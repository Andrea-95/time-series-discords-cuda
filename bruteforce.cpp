#include <iostream>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iterator>


using namespace std;


float sequencesDistance(int indexFirstSubsequence, int indexSecondSubsequence, int subSeqLength, float* timeSeries) {
    
    float distance = 0;
    float sum = 0;
    float diff = 0;
    float power = 1;
    int first = indexFirstSubsequence;
    int second = indexSecondSubsequence;

    for (int i = 0; i < subSeqLength; i++) {
        diff = timeSeries[first] - timeSeries[second];
        power = diff * diff;
        sum = sum + power;
        first++;
        second++;
    }
 
    distance = sqrt(sum);
    return distance;
}


void compareSubsequences(float* timeSeries, float* distances, int* locations, int subSeqLength, int tsLength) {

    float totalDist = 0;
    int loc = 0;

    for (int i = 0; i <= tsLength - subSeqLength * 2; i++) {                              // Outer loop
        for (int j = i; j <= tsLength - subSeqLength; j++) {                              // Inner loop
            if (abs(i - j) >= subSeqLength) {                                             // Non self match test
                totalDist = sequencesDistance(i, j, subSeqLength, timeSeries);

                if (totalDist < distances[i]) {
                    distances[i] = totalDist;
                    locations[i] = j;
                }

                if (totalDist < distances[j]) {
                    distances[j] = totalDist;
                    locations[j] = i;
                }
            }
        }
    }
}


//--------------------Lettura e scrittura file start--------------------------
void readFile(float* timeSeries, string fileName) {

    const char* c = fileName.c_str();
    float num = 0;
    int i = 0;
    ifstream readFile;
    readFile.open(c);

    if (!readFile.is_open()) {                                                       // Check to see that the file was opened correctly
        cerr << "There was a problem opening the input file!\n";
        exit(1);                                                                     // Exit or do additional error checking
    }

    while (readFile >> num) {                                                        // Keep storing values from the text file so long as data exists
        timeSeries[i] = float(num);
        i++;
    }

    readFile.close();
}


void scriviFile(float* distances, int* locations, string fileName, int tsLength, int subSeqLength) {
    
    FILE* fp;
    char nomeFile;
    //   nomeFile = "nnd.dat";

    fp = fopen("nnd.dat", "w+");
    for (int i = 0; i < tsLength - subSeqLength + 1; i++) {
        fprintf(fp, "%lf \n", distances[i]);
    }
    fclose(fp);

    //    nomeFile = "ngh"+fileName  +".dat";
    fp = fopen ( "loc.dat" , "w+" );
    for (int i=0; i< tsLength - subSeqLength + 1; i++) {
        fprintf ( fp , "%d \n " , locations[i] + 1);
    }
    fclose (fp) ;
    
    fp = fopen("total.dat", "w+");
    for (int i = 0; i < tsLength - subSeqLength + 1; i++) {
        fprintf(fp, "%lf \t %d \n", distances[i], locations[i] + 1);
    }
    fclose(fp);
}
//--------------------Lettura e scrittura file end-----------------------------


/*
void writeToFile(vector<float> &distances, vector<int> &locations, string fileName) {

    string nomeFile;
    nomeFile = ( "distances_" + fileName.erase(fileName.length() - 4, fileName.length()) + ".txt" ).c_str();
    ofstream output_distances(nomeFile);


    ofstream output_distances("distances_" + fileName.erase(fileName.length() - 4, fileName.length()) + ".txt");
    ostream_iterator<float> output_iterator1(output_distances, "\n");
    copy(distances.begin(), distances.end(), output_iterator1);

    ofstream output_locations("positions_" + fileName.erase(fileName.length() - 4, fileName.length()) + ".txt");
    ostream_iterator<float> output_iterator2(output_locations, "\n");
    copy(locations.begin(), locations.end(), output_iterator2);
}
*/


int main() {

    //string fileName = "ecg0606_1.csv";    // parametri csv
    //const int tsLength = 2299;

    string fileName = "nprs44.txt";         // parametri txt
    const int tsLength = 24125;

    //string fileName = "318_signal1.txt";         // parametri txt
    //const int tsLength = 586086;

    int subSeqLength = 200;

    float* timeSeries;
    float* distances;
    int* locations;

    timeSeries = (float*)malloc(tsLength * sizeof(float));
    distances = (float*)malloc(tsLength * sizeof(float));
    locations = (int*)malloc(tsLength * sizeof(int));

    fill_n(distances, tsLength, FLT_MAX);

    cout << "File name: " << fileName << endl;
    cout << "File length: " << tsLength << endl;
    cout << "Subsequence length: " << subSeqLength << endl;

    readFile(timeSeries, fileName);
    compareSubsequences(timeSeries, distances, locations, subSeqLength, tsLength);

    scriviFile(distances, locations, fileName, tsLength, subSeqLength);

    free(timeSeries);
    free(distances);
    free(locations);

    return 0;
}
