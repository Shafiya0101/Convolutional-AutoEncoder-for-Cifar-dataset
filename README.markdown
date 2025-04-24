# Convolutional AutoEncoder for CIFAR-100 Dataset

## Project Overview
This project implements a Convolutional AutoEncoder for image compression using the CIFAR-100 dataset. The autoencoder is designed to compress images into a lower-dimensional latent space and reconstruct them with minimal loss of quality. The notebook explores different architectures, hyperparameters, and evaluates performance using metrics like Peak Signal-to-Noise Ratio (PSNR) and compression rate.

## Features
- **Dataset**: CIFAR-100 dataset with 32x32 RGB images.
- **Model**: Convolutional AutoEncoder with encoder and decoder components.
- **Architecture**: Utilizes Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, UpSampling2D, and Dense layers.
- **Hyperparameter Tuning**: Modified architecture with deeper layers (64, 128, 256 filters) and a latent space of 128 units.
- **Evaluation Metrics**:
  - Compression rate: 83.33%
  - Average PSNR: 23.65 dB
- **Training**: Includes EarlyStopping, Adam optimizer, and Mean Squared Error (MSE) loss.

## Repository Structure
- `Convolutional_AutoEncoder_for_Cifar_dataset.ipynb`: Jupyter Notebook containing the full implementation, including data preprocessing, model definition, training, and evaluation.
- `README.md`: This file, providing an overview and instructions for the project.

## Requirements
To run the notebook, you need the following dependencies:
- Python 3.6+
- TensorFlow/Keras
- NumPy
- Matplotlib (for visualization, if needed)

You can install the required packages using:
```bash
pip install tensorflow numpy matplotlib
```

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/convolutional-autoencoder-cifar100.git
   cd convolutional-autoencoder-cifar100
   ```

2. **Open the Notebook**:
   Use Jupyter Notebook or JupyterLab to open `Convolutional_AutoEncoder_for_Cifar_dataset.ipynb`.

3. **Run the Notebook**:
   - Execute the cells sequentially to load the CIFAR-100 dataset, preprocess the data, define and train the autoencoder, and evaluate its performance.
   - Ensure you have a compatible GPU for faster training (optional but recommended).

4. **Evaluate Results**:
   - The notebook calculates the compression rate and PSNR for the test set.
   - Visualizations of original and reconstructed images can be generated using the `showOrigDec` function (requires Matplotlib).

## Results
- **Compression Rate**: Achieves 83.33% compression, significantly reducing the data size.
- **PSNR**: Average PSNR of 23.65 dB, indicating good reconstruction quality.
- **Conclusion**: The modified architecture with deeper layers and a smaller latent space outperforms the initial model, providing a solid foundation for image compression tasks. Further improvements could focus on optimizing PSNR and exploring advanced architectures.

## Future Improvements
- Enhance PSNR by experimenting with additional layers or alternative loss functions (e.g., SSIM-based loss).
- Include realistic compression overhead for practical deployment.
- Explore advanced architectures like Variational AutoEncoders (VAEs) or attention-based models.
- Add visualizations for training loss curves and more extensive image comparisons.

## Contributing
Contributions are welcome! Please feel free to:
- Open issues for bugs or feature requests.
- Submit pull requests with improvements or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The CIFAR-100 dataset is provided by the University of Toronto.
- Built using TensorFlow/Keras for deep learning.