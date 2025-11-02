#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

// =====================================================
// Kernel CUDA — Espelhamento Horizontal com dim3 - 2D
// =====================================================
__global__ void flipHorizontal(unsigned char *img, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Troca apenas a metade esquerda com a metade direita
    if (x < width / 2 && y < height)
    {
        int idx1 = (y * width + x) * 3;
        int idx2 = (y * width + (width - 1 - x)) * 3;

        for (int c = 0; c < 3; c++)
        {
            unsigned char temp = img[idx1 + c];
            img[idx1 + c] = img[idx2 + c];
            img[idx2 + c] = temp;
        }
    }
}

// =====================================================
// Função para carregar imagem PPM (formato P6)
// =====================================================
bool loadPPM(const char *filename, unsigned char **data, int &width, int &height)
{
    ifstream file(filename, ios::binary);
    if (!file)
    {
        cerr << "Erro ao abrir arquivo " << filename << endl;
        return false;
    }

    string header;
    file >> header; // deve ser "P6"
    if (header != "P6")
    {
        cerr << "Formato PPM inválido\n";
        return false;
    }

    file >> width >> height;
    int maxval;
    file >> maxval;
    file.ignore(1); // ignora o '\n'

    int size = width * height * 3; // RGB
    *data = new unsigned char[size];
    file.read(reinterpret_cast<char *>(*data), size);
    file.close();
    return true;
}

// =====================================================
// Função para salvar imagem PPM
// =====================================================
void savePPM(const char *filename, unsigned char *data, int width, int height)
{
    ofstream file(filename, ios::binary);
    file << "P6\n"
         << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char *>(data), width * height * 3);
    file.close();
}

// =====================================================
// Programa principal
// =====================================================
int main(int argc, char *argv[])
{
    unsigned char *h_img;
    int width, height;

    if (argc < 2)
    {
        cout << "Uso: " << argv[0] << " <input_image.ppm>" << endl;
        return -1;
    }

    if (!loadPPM(argv[1], &h_img, width, height))
        return -1;

    int img_size = width * height * 3;
    unsigned char *d_img;

    cudaMalloc(&d_img, img_size);
    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);

    // --- Grid 2D para mapear pixels ---
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // --- Chamada do kernel ---
    flipHorizontal<<<grid, block>>>(d_img, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

    savePPM("output.ppm", h_img, width, height);

    cout << "Imagem espelhada salva como output_flip.ppm" << endl;

    cudaFree(d_img);
    delete[] h_img;
    return 0;
}