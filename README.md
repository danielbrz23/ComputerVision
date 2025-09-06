# Computer Vision - Criação de Panoramas com Image Stitching

Este repositório reúne materiais, códigos e projetos da matéria de Visão Computacional da UNICAMP. O foco principal está em técnicas de processamento de imagens, extração de características, montagem de panoramas e aplicação de algoritmos clássicos da área.

## Estrutura do Repositório

- **Image Stitching/**  
  Diretório dedicado ao projeto de montagem de panoramas a partir de múltiplas imagens.
  - `Panoram.ipynb`: Notebook explicando e demonstrando o pipeline de criação de panoramas.
  - `model.py`: Implementação dos principais algoritmos usados para detecção, descrição e correspondência de pontos característicos entre imagens.
  - Imagens de exemplo:
    - `matches_placa.jpg`, `orb_descriptor.jpg`, `result_IFCH.jpg`, `result_placa.jpg`, `sift_descriptor.jpg`
  - `data/`: Pasta destinada a arquivos de dados para testes.

- **relatorio/**  
  Pasta para relatórios e documentação complementar sobre os experimentos realizados.

- **.gitignore**  
  Arquivo de configuração para ignorar arquivos temporários e de ambiente.

## Principais Funcionalidades

- Detecção e descrição de pontos de interesse utilizando algoritmos SIFT e ORB.
- Correspondência de pontos entre imagens para criação de mosaicos/panoramas.
- Geração dos resultados intermediários e finais em arquivos de imagem.
- Documentação e exemplos em notebook para facilitar o entendimento dos métodos aplicados.
