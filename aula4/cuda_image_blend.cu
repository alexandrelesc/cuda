#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

// =====================================================
// Kernel CUDA — Mistura (blend) de duas imagens
// =====================================================
__global__ void blendImages(unsigned char *imgA, unsigned char *imgB, unsigned char *imgOut, int size, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        imgOut[i] = alpha * imgA[i] + (1.0f - alpha) * imgB[i];
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
    unsigned char *h_imgA, *h_imgB;
    int widthA, heightA, widthB, heightB;

    if (argc < 3)
    {
        cout << "Uso: " << argv[0] << " <imagem1.ppm> <imagem2.ppm>" << endl;
        return -1;
    }

    if (!loadPPM(argv[1], &h_imgA, widthA, heightA))
        return -1;

    if (!loadPPM(argv[2], &h_imgB, widthB, heightB))
        return -1;

    // Verifica se as imagens têm o mesmo tamanho
    if (widthA != widthB || heightA != heightB)
    {
        cout << "As imagens devem ter o mesmo tamanho!" << endl;
        delete[] h_imgA;
        delete[] h_imgB;
        return -1;
    }

    int width = widthA;
    int height = heightA;
    int img_size = width * height * 3;

    unsigned char *d_imgA, *d_imgB, *d_imgOut;
    cudaMalloc(&d_imgA, img_size);
    cudaMalloc(&d_imgB, img_size);
    cudaMalloc(&d_imgOut, img_size);

    cudaMemcpy(d_imgA, h_imgA, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgB, h_imgB, img_size, cudaMemcpyHostToDevice);

    // --- Configuração de threads e blocos ---
    int threads = 256;
    int blocks = (img_size + threads - 1) / threads;

    // --- Fator de mistura (50% de cada imagem) ---
    float alpha = 0.5f;

    // --- Chamada do kernel ---
    blendImages<<<blocks, threads>>>(d_imgA, d_imgB, d_imgOut, img_size, alpha);
    cudaDeviceSynchronize();

    // --- Copia o resultado para a CPU ---
    unsigned char *h_imgOut = new unsigned char[img_size];
    cudaMemcpy(h_imgOut, d_imgOut, img_size, cudaMemcpyDeviceToHost);

    // --- Salva imagem resultante ---
    savePPM("output_blend.ppm", h_imgOut, width, height);

    cout << "Imagem mesclada salva como output_blend.ppm" << endl;

    cudaFree(d_imgA);
    cudaFree(d_imgB);
    cudaFree(d_imgOut);
    delete[] h_imgA;
    delete[] h_imgB;
    delete[] h_imgOut;

    return 0;
}