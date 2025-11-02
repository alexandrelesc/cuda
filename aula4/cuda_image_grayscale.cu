#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

// =====================================================
// Kernel CUDA — Conversão para tons de ciza da imagem
// =====================================================
__global__ void grayscale(unsigned char *img, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 3;
    if (idx + 2 < size)
    {
        unsigned char r = img[idx];
        unsigned char g = img[idx + 1];
        unsigned char b = img[idx + 2];
        unsigned char gray = 0.299f * r + 0.587f * g + 0.114f * b;
        img[idx] = img[idx + 1] = img[idx + 2] = gray;
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

    if(argc < 2) {
        cout << "Uso: " << argv[0] << "./app <path para input_image.ppm>" << endl;
        return -1;
    }
    
    if (!loadPPM(argv[1], &h_img, width, height))
        return -1;

    int img_size = width * height * 3;

    unsigned char *d_img;
    cudaMalloc(&d_img, img_size);

    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (img_size + threads - 1) / threads;

    grayscale<<<blocks, threads>>>(d_img, img_size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

    savePPM("output.ppm", h_img, width, height);

    cout << "Imagem processada e salva como output.ppm" << endl;

    cudaFree(d_img);
    delete[] h_img;
    return 0;
}
