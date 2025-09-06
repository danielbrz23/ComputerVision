import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
import warnings
import os
import matplotlib.pyplot as plt
import imutils
from PIL import Image
import pillow_heif

def heif2jpg(heif_img:str):
    heif_file = pillow_heif.read_heif(heif_img)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw"
    )
    img = np.array(image)
    img = cv2.resize(img, (0,0),fx=.5, fy=.5)
    return img

class PanoramaStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=1000)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=25)  # Aumentar para mais precisão, diminuir para mais velocidade

        self.bf = cv2.FlannBasedMatcher(index_params, search_params)
        self.images = []
        self.features = []
        self.matches = []
        self.homographies = []
        self.focal_lengths = []
        self.rotations = []
        self.gains = []

    def extract_features(self, image):
        """Extract SIFT features from an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)
        return {'keypoints': kp, 'descriptors': desc}

    def match_features_flann(self, desc1, desc2):
        """
        Encontra os melhores matches entre dois descritores usando FLANN.
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        
        # Usa knnMatch para encontrar os 2 vizinhos mais próximos de forma eficiente
        matches = self.bf.knnMatch(desc1, desc2, k=4)

        # Aplica o teste de proporção de Lowe para filtrar matches ruins
        good_matches = []
        for match_list in matches:
            # Garante que temos um par de matches para comparar
            if len(match_list) >= 2:
                m, n = match_list[0], match_list[1] # Extrai o primeiro e o segundo vizinho
                if m.distance < 0.60 * n.distance:
                    good_matches.append(m)
        
        return good_matches

    def ransac_homography(self, src_pts, dst_pts, n_trials=500, threshold=5.0):
        """Estimate homography using RANSAC with probabilistic verification"""
        if len(src_pts) < 4:
            return None, None

        # Convert points to numpy arrays
        src_pts = np.float32(src_pts).reshape(-1, 2)
        dst_pts = np.float32(dst_pts).reshape(-1, 2)

        # RANSAC parameters
        best_H = None
        best_inliers = []
        max_inliers = 0

        for _ in range(n_trials):
            # Randomly select 4 point pairs
            idx = np.random.choice(len(src_pts), 4, replace=False)
            src_sample = src_pts[idx]
            dst_sample = dst_pts[idx]

            # Compute homography
            H, _ = cv2.findHomography(src_sample, dst_sample, 0)
            if H is None:
                continue

            # Transform all points
            ones = np.ones((len(src_pts), 1))
            src_hom = np.hstack((src_pts, ones))
            dst_proj = H @ src_hom.T
            dst_proj = (dst_proj[:2] / dst_proj[2]).T

            # Compute distances
            errors = np.linalg.norm(dst_pts - dst_proj, axis=1)
            inliers = errors < threshold

            # Update best model
            inlier_count = np.sum(inliers)
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_inliers = inliers
                best_H = H

        # Probabilistic verification
        n_inliers = max_inliers
        n_total = len(src_pts)
        alpha = 8.0
        beta = 0.3

        if best_H is not None and np.sum(best_inliers) >= 4 and n_inliers > alpha + beta * n_total:
            # Refine homography with all inliers
            src_inliers = src_pts[best_inliers]
            dst_inliers = dst_pts[best_inliers]
            best_H, _ = cv2.findHomography(src_inliers, dst_inliers, cv2.RANSAC, threshold)
            return best_H, best_inliers
        else:
            return None, None

    def bundle_adjustment(self):
        """Joint optimization of all camera parameters"""
        if len(self.images) < 2:
            return

        # Initialize parameters: each image has rotation (3 params) and focal length (1 param)
        n_images = len(self.images)
        params = np.zeros(n_images * 4)

        # Set initial values
        for i in range(n_images):
            if i < len(self.rotations):
                rvec, _ = cv2.Rodrigues(self.rotations[i])
                params[i*4:i*4+3] = rvec.flatten()
            if i < len(self.focal_lengths):
                params[i*4+3] = self.focal_lengths[i]
            else:
                params[i*4+3] = self.focal_lengths[0] if self.focal_lengths else 1000.0

        # Optimize using Levenberg-Marquardt
        def residual(params):
            rotations = []
            focal_lengths = []
            for i in range(n_images):
                theta = params[i*4:i*4+3]
                R = cv2.Rodrigues(theta)[0]
                rotations.append(R)
                focal_lengths.append(params[i*4+3])

            residuals = []
            for match in self.matches:
                img_idx1, img_idx2 = match['image_indices']
                kp_idx1, kp_idx2 = match['keypoint_indices']

                kp1 = self.features[img_idx1]['keypoints'][kp_idx1].pt
                kp2 = self.features[img_idx2]['keypoints'][kp_idx2].pt

                K1 = np.array([
                    [focal_lengths[img_idx1], 0, 0],
                    [0, focal_lengths[img_idx1], 0],
                    [0, 0, 1]
                ])
                K2 = np.array([
                    [focal_lengths[img_idx2], 0, 0],
                    [0, focal_lengths[img_idx2], 0],
                    [0, 0, 1]
                ])

                H = K1 @ rotations[img_idx1] @ rotations[img_idx2].T @ np.linalg.inv(K2)
                point = np.array([kp2[0], kp2[1], 1])
                projected = H @ point
                projected = projected[:2] / projected[2]
                residual = np.array([kp1[0] - projected[0], kp1[1] - projected[1]])
                residuals.append(residual)

            return np.concatenate(residuals)

        # Run optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = least_squares(residual, params, method='lm', max_nfev=100)

        # Update parameters
        for i in range(n_images):
            theta = result.x[i*4:i*4+3]
            self.rotations[i] = cv2.Rodrigues(theta)[0]
            self.focal_lengths[i] = result.x[i*4+3]

    def normalize_exposure(self, images):
        """Normalize exposure across images before stitching"""
        lab_images = [cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for img in images]

        # Calculate median L values (more robust than mean)
        l_medians = [np.median(img[:,:,0]) for img in lab_images]
        ref_median = np.median(l_medians)

        # Adjust each image
        normalized = []
        for img, l_median in zip(lab_images, l_medians):
            if l_median > 0:
                scale = ref_median / l_median
                # Apply more conservative scaling
                scale = np.clip(scale, 0.8, 1.2)
                img[:,:,0] = np.clip(img[:,:,0] * scale, 0, 255)
            normalized.append(cv2.cvtColor(img, cv2.COLOR_LAB2BGR))

        return normalized

    def gain_compensation(self):
        """More conservative gain compensation"""
        n_images = len(self.images)
        if n_images < 2:
            self.gains = np.ones(n_images)
            return

        # Calculate median intensities in LAB color space
        median_intensities = []
        for img in self.images:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            median_intensities.append(np.median(lab[:,:,0]))

        # Use median intensity as reference
        reference_intensity = np.median(median_intensities)

        # Calculate gains relative to reference
        self.gains = []
        for intensity in median_intensities:
            if intensity > 0:
                gain = reference_intensity / intensity
                # Apply conservative clamping
                gain = np.clip(gain, 0.7, 1.3)
                self.gains.append(gain)
            else:
                self.gains.append(1.0)

    def apply_gain_compensation(self, img, gain):
        """Apply gain compensation with gamma correction for more natural results"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0].astype(np.float32)

        # Apply gain with gamma correction
        adjusted = 255 * (l_channel/255) ** (1/gain)

        lab[:,:,0] = np.clip(adjusted, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    def straighten_panorama(self):
        """
        Endireita o panorama calculando uma rotação global de correção.
        Esta técnica baseia-se na heurística de que os eixos X das câmaras
        tendem a situar-se num plano comum.
        """
        print("Straightening panorama...")

        # 1. Obter todos os vetores X da câmara (primeira coluna de cada matriz de rotação)
        camera_x_vectors = []
        for r in self.rotations:
            # O vetor X é a primeira coluna da matriz de rotação
            camera_x_vectors.append(r[:, 0])
        
        camera_x_vectors = np.array(camera_x_vectors)

        # 2. Calcular a matriz de covariância
        # C = X^T * X
        covariance_matrix = camera_x_vectors.T @ camera_x_vectors

        # 3. Encontrar o autovetor associado ao menor autovalor
        # Este é o nosso "vetor para cima" (u), que é normal ao plano dos vetores X.
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        up_vector = eigen_vectors[:, np.argmin(eigen_values)]

        # 4. Calcular a rotação para alinhar o nosso 'up_vector' com o eixo Y vertical [0, 1, 0]
        # O eixo de rotação é o produto vetorial entre os dois vetores
        rotation_axis = np.cross(up_vector, [0, 1, 0])
        rotation_axis /= np.linalg.norm(rotation_axis) # Normalizar

        # O ângulo de rotação é o arco-cosseno do produto escalar
        angle = np.arccos(np.dot(up_vector, [0, 1, 0]))

        # Criar a matriz de rotação global de correção usando a fórmula de Rodrigues
        # (que converte de eixo-ângulo para matriz de rotação)
        R_straight, _ = cv2.Rodrigues(rotation_axis * angle)

        # 5. Aplicar esta rotação de correção a todas as câmaras
        for i in range(len(self.rotations)):
            self.rotations[i] = R_straight @ self.rotations[i]
    
    def feathering(self, warped_images, masks, blend_width=20):
      """
      Mescla imagens usando feathering com largura de blend ajustável.

      Args:
          warped_images: Lista de imagens distorcidas.
          masks: Lista de máscaras correspondentes.
          blend_width: Largura da área de suavização (em pixels). 
                      Valores maiores criam um blend mais suave. Use um número par.
      """
      if not warped_images:
          return None

      canvas_size = warped_images[0].shape
      result = np.zeros(canvas_size, dtype=np.float32)
      total_weight = np.zeros(canvas_size, dtype=np.float32)

      # O kernel controla o quanto a máscara será "encolhida" (erodida)
      # Um kernel maior resulta em um blend mais largo
      kernel_size = blend_width + 1 if blend_width % 2 == 0 else blend_width # Garante que seja ímpar
      kernel = np.ones((kernel_size, kernel_size), np.uint8)

      for img, mask in zip(warped_images, masks):
          mask_uint8 = (mask * 255).astype(np.uint8)
          
          # Erodindo a máscara para criar uma zona de transição mais larga
          eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)

          # O peso é a distância da borda da máscara erodida
          weight = cv2.distanceTransform(eroded_mask, cv2.DIST_L2, 5).astype(np.float32)
          
          # Normalizar o peso para o intervalo [0, 1]
          max_val = np.max(weight)
          if max_val > 0:
              weight /= max_val
          
          weight = cv2.merge([weight] * 3) # Expandir para 3 canais

          result += img.astype(np.float32) * weight
          total_weight += weight

      # Evitar divisão por zero
      mask_div = total_weight > 1e-6
      blended_panorama = np.zeros_like(result, dtype=np.float32)
      np.divide(result, total_weight, out=blended_panorama, where=mask_div)

      return np.clip(blended_panorama, 0, 255).astype(np.uint8)

    def adjust_tone(self, image):
        """Final tone adjustment with more conservative settings"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE with conservative settings
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        l = clahe.apply(l)

        # Apply mild gamma correction
        l = np.power(l.astype(np.float32)/255, 0.95) * 255
        l = np.clip(l, 0, 255).astype(np.uint8)

        # Merge channels back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def auto_crop(self, image):
        """Crop black borders from the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return image

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        return image[y:y+h, x:x+w]
    
    def stitch(self, images):
        """Main stitching pipeline with improved exposure handling"""
        # Normalize exposure first
        self.images = self.normalize_exposure(images)
        n_images = len(self.images)

        if n_images == 0:
            return None

        # Step 1: Feature extraction
        print("Extracting features...")
        self.features = [self.extract_features(img) for img in self.images]

        # Step 2: Feature matching
        print("Matching features...")
        self.matches = []
        for i in range(n_images):
            for j in range(i+1, n_images):
                matches = self.match_features_flann(
                    self.features[i]['descriptors'],
                    self.features[j]['descriptors']
                )

                if len(matches) < 4:
                    continue

                src_pts = np.float32([self.features[i]['keypoints'][m.queryIdx].pt for m in matches])
                dst_pts = np.float32([self.features[j]['keypoints'][m.trainIdx].pt for m in matches])

                #changing threshold to 3
                H, inliers = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1.5)

                if H is not None:
                    inlier_matches = [matches[k] for k in range(len(matches)) if inliers[k]]
                    for m in inlier_matches:
                        self.matches.append({
                            'image_indices': (i, j),
                            'keypoint_indices': (m.queryIdx, m.trainIdx)
                        })

        # Step 3: Initialize camera parameters
        print("Initializing camera parameters...")
        self.focal_lengths = [1000.0] * n_images
        self.rotations = [np.eye(3) for _ in range(n_images)]

        # Step 4: Bundle adjustment
        print("Running bundle adjustment...")
        self.bundle_adjustment()
        self.straighten_panorama()
        # Step 5: Gain compensation
        print("Computing gain compensation...")
        self.gain_compensation()
        print("Computed gains:", self.gains)

        # Step 6: Warp images and blend
        print("Warping and blending images...")

        ref_idx = n_images // 2
        h_ref, w_ref = self.images[ref_idx].shape[:2]

        K_ref = np.array([
            [self.focal_lengths[ref_idx], 0, 0],
            [0, self.focal_lengths[ref_idx], 0],
            [0, 0, 1]
        ])

        # Compute panorama bounding box
        corners = []
        for i in range(n_images):
            h_i, w_i = self.images[i].shape[:2]
            pts = np.array([[0, 0], [w_i, 0], [w_i, h_i], [0, h_i]], dtype=np.float32).reshape(-1, 1, 2)

            K_i = np.array([
                [self.focal_lengths[i], 0, 0],
                [0, self.focal_lengths[i], 0],
                [0, 0, 1]
            ])
            H = K_ref @ self.rotations[ref_idx] @ self.rotations[i].T @ np.linalg.inv(K_i)

            warped_corners = cv2.perspectiveTransform(pts, H)
            corners.append(warped_corners)

        all_corners = np.vstack(corners)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        canvas_size = (x_max - x_min, y_max - y_min)

        # Warp all images with compensation
        warped_images = []
        masks = []
        for i in range(n_images):
            K_i = np.array([
                [self.focal_lengths[i], 0, 0],
                [0, self.focal_lengths[i], 0],
                [0, 0, 1]
            ])
            H = K_ref @ self.rotations[ref_idx] @ self.rotations[i].T @ np.linalg.inv(K_i)

            img = self.apply_gain_compensation(self.images[i], self.gains[i])
            warped = cv2.warpPerspective(img, shift @ H, canvas_size)
            warped_images.append(warped)

            mask = np.ones((self.images[i].shape[0], self.images[i].shape[1]), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, shift @ H, canvas_size)
            masks.append((warped_mask > 0).astype(np.float32))

        # Use Feather Blending instead of Multi-band
        panorama = self.feathering(warped_images, masks, 20)

        # Optional post-processing
        # panorama = self.adjust_tone(panorama)
        # panorama = self.auto_crop(panorama)

        return panorama
    
if __name__ == '__main__':
    
    path = '/home/danbrz/Projects/ComputerVision/Image Stitching/data/raw/scene3'

    scene_path = [os.path.join(path, file) for file in sorted(os.listdir(path))]

    stitcher = PanoramaStitcher()
    imgs = [cv2.resize(heif2jpg(p), (0, 0), fx=1, fy=1) for p in scene_path]
    panorama = stitcher.stitch(imgs[:12])

    if panorama is not None:
        plt.imshow(panorama)
        cv2.imwrite( )
        cv2.imwrite('result_placa.jpg', cv2.cvtColor(stitcher.adjust_tone(panorama), cv2.COLOR_RGB2BGR))