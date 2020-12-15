/*
#ifndef __CUDACC__
#define __CUDACC__
#endif
*/

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iterator>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>

#define SUBSEQLENGTH 200
#define NUMTHREADS 1024
#define TSLENGTH 24125                      // Lunghezza txt
#define NUMOFSUBSEQ (TSLENGTH - SUBSEQLENGTH + 1)
#define SHAREDLENGTH (NUMTHREADS + SUBSEQLENGTH - 1)
//#define TSLENGTH 2299                     // Lunghezza csv
//#define TSLENGTH 586086

#define NUMBLOCKS (TSLENGTH + NUMTHREADS - 1) / NUMTHREADS          // Calcolo del numero ottimale di blocchi

__constant__ float primo_vettore_confronto[SUBSEQLENGTH];          // Si crea il vettore della memoria constant e lo si riempe con una parte
                                                                    // del vettore timeSeries lungo SUBSEQLENGTH partendo dall'indice i

using namespace std;


__inline__ __device__ void warpReduceMin(float& val, int& idx) {

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {

        float tmpVal = __shfl_down_sync(0xFFFFFFFF, val, offset, 32);
        int tmpIdx = __shfl_down_sync(0xFFFFFFFF, idx, offset, 32);

        if (tmpVal == val) {                                        // Se due valori di distanza sono uguai si salva la posizione più piccola
            if (tmpIdx < idx) {
                idx = tmpIdx;
            }
        }

        if (tmpVal < val) {
            val = tmpVal;
            idx = tmpIdx;
        }
    }
}


__inline__ __device__ void blockReduceMin(float& val, int& idx, int currentThreads, int indexFirstSubsequence) {

    static __shared__ float values[32], indices[32];            // Shared mem for 32 partial mins

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    if (lane == 0) {                                            // Il primo thread di ogni warp inizializza un elemento della memoria shared
        values[wid] = FLT_MAX;                                  // all'indice che corrisponde a quello del blocco di cui fa parte
    }

    warpReduceMin(val, idx);                                     // Each warp performs partial reduction

    if (lane == 0) {
        values[wid] = val;                                       // Write reduced value to shared memory
        indices[wid] = idx;                                      // Write reduced value to shared memory
    }

    __syncthreads();                                             // Wait for all partial reductions

    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) {  // Read from shared memory only if that warp existed
        val = values[lane];                                      // && threadIdx.x <= currentThreads / warpSize
        idx = indices[lane];
    }
    else {
        val = FLT_MAX;
        idx = 0;
    }

    if (wid == 0) {
        warpReduceMin(val, idx);                                 // Final reduce within first warp
    }
}


__global__ void sequencesDistance(int indexFirstSubsequence, float* dev_timeSeries, float* dev_blocksDistances, int* dev_blocksLocations) {

    extern __shared__ float cache[];                                  // La sua lunghezza è NUMTHREADS + SUBSEQLENGTH - 1
    float sum = 0;
    float distanza = FLT_MAX;                                         // Variabile a cui è assegnata la distanza. I thread non coinvolti nel calcolo hanno questo valore come default
    float diff;
    float power;

    int first_subseq_idx = indexFirstSubsequence;                       // Indice della prima sottosequenza che verrà confrontata con tutte le altre
    int second_subseq_idx = blockIdx.x * blockDim.x + threadIdx.x;      // Indice della seconda sottosequenza che si confronta con la prima. L'id globale di ogni thread stabilisce il punto di partenza
    int dev_ts_position = blockIdx.x * blockDim.x + threadIdx.x;
    int cache_idx = threadIdx.x;


    while (cache_idx < SHAREDLENGTH && dev_ts_position <= TSLENGTH) {  // Ogni thread carica nella shared uno o più elementi di timeSeries
        cache[cache_idx] = dev_timeSeries[dev_ts_position];
        cache_idx = cache_idx + NUMTHREADS;
        dev_ts_position = dev_ts_position + NUMTHREADS;
    }

    __syncthreads();                                                   // SYNC perché non tutti possono aver già riempito la cache

    if (abs(second_subseq_idx - first_subseq_idx) >= SUBSEQLENGTH && second_subseq_idx < NUMOFSUBSEQ) {   // Verifica di self-match e controllo che il secondo indice rientri nei valori consentiti
        for (int i = 0; i < SUBSEQLENGTH; i++) {
            diff = primo_vettore_confronto[i] - cache[threadIdx.x + i];
            power = diff * diff;
            sum = sum + power;
        }
        distanza = sqrt(sum);                                          // Da rimuovere nella versione finale, la radice si calcola alla fine
    }

    blockReduceMin(distanza, second_subseq_idx, NUMTHREADS, first_subseq_idx);

    if (threadIdx.x == 0) {
        dev_blocksDistances[blockIdx.x] = distanza;
        dev_blocksLocations[blockIdx.x] = second_subseq_idx;
    }
}


__global__ void finalReduction(int indexFirstSubsequence, int previousBlocks, int* dev_blocksLocations, float* dev_blocksDistances, int* dev_finalLocations, float* dev_finalDistances) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = FLT_MAX;
    int idx = 0;

    if (tid < previousBlocks) {
        val = dev_blocksDistances[tid];
        idx = dev_blocksLocations[tid];
    }

    blockReduceMin(val, idx, previousBlocks, indexFirstSubsequence);

    if (threadIdx.x == 0 && gridDim.x != 1) {   // Si utilizzano i vettori dev_blocksDistances e dev_blocksLocations per salvare i risultati delle riduzioni
        dev_blocksDistances[blockIdx.x] = val;  // ad ogni nuova iterazione finché si utilizza più di un blocco
        dev_blocksLocations[blockIdx.x] = idx;
    }

    if (tid == 0 && gridDim.x == 1) {           // Si scrive sui vettori finali solo quando la riduzione è arrivata ad utilizzare un solo blocco
        dev_finalDistances[indexFirstSubsequence] = val;
        dev_finalLocations[indexFirstSubsequence] = idx;
    }
}


void compareSubsequences(float* dev_blocksDistances, int* dev_blocksLocations, float* dev_timeSeries, float* dev_finalDistances, int* dev_finalLocations) {

    int threads = 1024;

    for (int i = 0; i <= TSLENGTH - SUBSEQLENGTH; i++) {        // Outer loop

        int previousBlocks = NUMBLOCKS;
        int currentBlocks = (NUMBLOCKS + threads - 1) / threads;

        bool continueReduction = true;

        cudaMemcpyToSymbol(primo_vettore_confronto, &dev_timeSeries[i], SUBSEQLENGTH * sizeof(float), 0, cudaMemcpyDeviceToDevice);  // Copia nella constant la sottosequenza all'i-esima posizione
                                                                                                                                      // da confrontare con tutte le altre

        sequencesDistance<<<NUMBLOCKS, NUMTHREADS, SHAREDLENGTH * sizeof(float)>>>(i, dev_timeSeries, dev_blocksDistances, dev_blocksLocations);  // Kernel che esegue il calcolo delle distanze
                                                                                                                                                   // ed una prima riduzione

        while (continueReduction) {                            // Si continua a ridurre i risultati finché il secondo kernel non ha un solo blocco
            finalReduction<<<currentBlocks, threads>>>(i, previousBlocks, dev_blocksLocations, dev_blocksDistances, dev_finalLocations, dev_finalDistances); // Riduce i risultati ottenuti dal kernel precedente

            if (currentBlocks == 1) {
                continueReduction = false;
            }

            previousBlocks = currentBlocks;
            currentBlocks = (currentBlocks + threads - 1) / threads;    // Calcolo del nuovo numero di blocchi da usare in finalReduction
        }
    }
    cudaFree(primo_vettore_confronto);
}


//--------------------Lettura e scrittura file start--------------------------
void readFile(float* timeSeries, string fileName) {

    const char* c = fileName.c_str();
    float num = 0;
    int i = 0;
    ifstream readFile;
    readFile.open(c);

    if (!readFile.is_open()) {                                                    // Check to see that the file was opened correctly
        cerr << "There was a problem opening the input file!\n";
        exit(1);                                                                  // Exit or do additional error checking
    }

    while (readFile >> num) {                                                     // Keep storing values from the text file so long as data exists
        timeSeries[i] = float(num);
        i++;
    }

    readFile.close();
}


void scriviFile(float* distances, int* locations, string fileName) {

    FILE* fp;
    //char nomeFile;
    //nomeFile = "nnd.dat";

    fp = fopen("nnd.dat", "w+");
    for (int i = 0; i < NUMOFSUBSEQ; i++) {
        fprintf(fp, "%lf \n", distances[i]);
    }
    fclose(fp);

    //    nomeFile = "ngh"+fileName  +".dat";
    fp = fopen("loc.dat", "w+");
    for (int i = 0; i < NUMOFSUBSEQ; i++) {
        fprintf(fp, "%d \n ", locations[i] + 1);
    }
    fclose(fp);

    fp = fopen("total.dat", "w+");
    for (int i = 0; i < NUMOFSUBSEQ; i++) {
        fprintf(fp, "%lf \t %d \n", distances[i], locations[i] + 1);
    }
    fclose(fp);
}
//--------------------Lettura e scrittura file end-----------------------------


int main() {

    //string fileName = "ecg0606_1.csv";
    string fileName = "nprs44.txt";
    //string fileName = "318_signal1.txt";

    float* timeSeries;
    float* distances;
    int* locations;

    timeSeries = (float*)malloc(TSLENGTH * sizeof(float));
    distances = (float*)malloc(NUMOFSUBSEQ * sizeof(float));
    locations = (int*)malloc(NUMOFSUBSEQ * sizeof(int));

    fill_n(distances, NUMOFSUBSEQ, FLT_MAX);

    cout << "File name: " << fileName << endl;
    cout << "File length: " << TSLENGTH << endl;
    cout << "Subsequence length: " << SUBSEQLENGTH << endl;
    cout << "Number of blocks: " << NUMBLOCKS << endl;
    cout << "Number of threads: " << NUMTHREADS << endl;
    cout << "Number of subsequences: " << NUMOFSUBSEQ << endl;
    cout << "Shared memory dimension: " << SHAREDLENGTH << endl;

    readFile(timeSeries, fileName);

    float* dev_blocksDistances;                 // Vettore sulla memoria global dove ogni blocco salva il risultato della distanza che calcola
    int* dev_blocksLocations;                    // Vettore sulla memoria global dove ogni blocco salva l'indice della propria migliore distanza trovata
    float* dev_finalDistances;
    int* dev_finalLocations;
    float* dev_timeSeries;

    cudaMalloc((void**)&dev_blocksDistances, NUMBLOCKS * sizeof(float));
    cudaMalloc((void**)&dev_blocksLocations, NUMBLOCKS * sizeof(int));
    cudaMalloc((void**)&dev_timeSeries, TSLENGTH * sizeof(float));
    cudaMalloc((void**)&dev_finalDistances, NUMOFSUBSEQ * sizeof(float));
    cudaMalloc((void**)&dev_finalLocations, NUMOFSUBSEQ * sizeof(int));

    cudaMemcpy(dev_timeSeries, timeSeries, TSLENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_finalDistances, distances, NUMOFSUBSEQ * sizeof(float), cudaMemcpyHostToDevice);

    compareSubsequences(dev_blocksDistances, dev_blocksLocations, dev_timeSeries, dev_finalDistances, dev_finalLocations);     // Funzione che lancia il kernel

    cudaMemcpy(distances, dev_finalDistances, NUMOFSUBSEQ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(locations, dev_finalLocations, NUMOFSUBSEQ * sizeof(int), cudaMemcpyDeviceToHost);

    scriviFile(distances, locations, fileName);

    free(timeSeries);
    free(distances);
    free(locations);

    cudaFree(dev_blocksDistances);
    cudaFree(dev_blocksLocations);
    cudaFree(dev_timeSeries);
    cudaFree(dev_finalDistances);
    cudaFree(dev_finalLocations);

    return 0;
}
