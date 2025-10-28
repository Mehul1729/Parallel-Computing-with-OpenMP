// code for fusing LidAr and Radar data using Serial, CPU and fake GPU execution:

#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // For fminf, sinf, cosf, logf, sqrtf  funcs 
#include <omp.h>
#include <time.h>   
#include <string.h> // Added for sprintf (to create filenames)

#define frames 1000
#define sectors 360
#define N (frames * sectors) // Total data points
#define noise_std 0.1f           // standard deviation for gaussian noise

// pi value  required for adding gaussian noise in C :
#ifndef M_PI
#define M_PI 3.14
#endif

// helper function to add gaussiuan noise to simulate real world sensor data :
float rand_normal(float mean, float stddev, unsigned int *seed) {

    // box muller transform to convert the two uniform RV to Gaussian 
    float u1 = (float)rand_r(seed) / (float)RAND_MAX;
    float u2 = (float)rand_r(seed) / (float)RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return z0 * stddev + mean;
}

// ======== Sensor data generation ===========
void generate_sensor_data(float *data, float base, float scale) {

    // Here I'm using a private seed for each thread to generate aensor data with random gaussian npise 
    unsigned int seed;
    #pragma omp parallel private(seed)
    {
        seed = (unsigned int)time(NULL) + (unsigned int)omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            // generating random values for given base and scale for a sensor :
            data[i] = base + (float)rand_r(&seed) / (float)RAND_MAX * scale;
        }
    }
}

// ============= Serial fusion version  =================
// baseline for comparison :
void fuse_serial(float *lidar, float *radar, float *fused) {
    unsigned int seed = (unsigned int)time(NULL); // Single seed for serial run
    for (int i = 0; i < N; i++) {

        fused[i] = fminf(lidar[i], radar[i]) + rand_normal(0.0f, noise_std, &seed);
    }
}


// ============= Openmp CPU fusion version  =================
void fuse_cpu(float *lidar, float *radar, float *fused) {
    unsigned int seed;
    #pragma omp parallel private(seed)
    {
        seed = (unsigned int)time(NULL) + (unsigned int)omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
           

            fused[i] = fminf(lidar[i], radar[i]) + rand_normal(0.0f, noise_std, &seed);
        }
    }
}


// ============ Fake GPU fusion ==============
// Modified to accept chunk_size as a parameter
void fake_gpu(float *lidar, float *radar, float *fused, int current_chunk_size) {
    unsigned int seed;
    #pragma omp parallel private(seed)
    {
        seed = (unsigned int)time(NULL) + (unsigned int)omp_get_thread_num();

        // Simulating GPU by parallelizing with fine-grained loop scheduling (dynamic )
        // Used the passed parameter 'current_chunk_size'
        #pragma omp for schedule(dynamic, current_chunk_size)
        for (int i = 0; i < N; i++) {


            fused[i] = fminf(lidar[i], radar[i]) + rand_normal(0.0f, noise_std, &seed);
            
            // Artificial delay to simulate gpu architecture :
            for (volatile int k = 0; k < 5; k++); 
        }
    }
}

// ============= MAIN FUNCTION ==============
int main() {
    static float *lidar, *radar, *fused;
    
    lidar = (float*)malloc(N * sizeof(float));
    radar = (float*)malloc(N * sizeof(float));
    fused = (float*)malloc(N * sizeof(float));



// Seeding the main random number generator:
    srand((unsigned int)time(NULL));

    printf("\n=== AUTONOMOUS VEHICLE SENSOR FUSION (Serial vs OpenMP vs GPU Sim) ===\n");
    printf("Total frames: %d | Sectors per frame: %d | Total points: %d\n", 
           frames, sectors, N);

// Data Initialization :
    double t0 = omp_get_wtime();
    generate_sensor_data(lidar, 40.0f, 20.0f); // LiDAR: 40m base, 20m range
    generate_sensor_data(radar, 30.0f, 40.0f); // Radar: 30m base, 40m range
    double t1 = omp_get_wtime();
    printf("Sensor data generated in %.4f sec\n\n", t1 - t0);

// Runining serial fusion :
    printf("Running Serial baseline...");
    fflush(stdout);
    double start_serial = omp_get_wtime();
    fuse_serial(lidar, radar, fused);
    double end_serial = omp_get_wtime();
    double serial_time = end_serial - start_serial;
    printf(" Done in %.4f sec\n\n", serial_time);

    // varying the chunk sizes for Fake GPU:
    int chunk_sizes[] = {4, 8, 18, 32, 64, 128, 256, 360, 512};
    int num_chunk_sizes = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    for (int c = 0; c < num_chunk_sizes; c++) {
        int current_chunk_size = chunk_sizes[c];
// saving rthe data :
        char filename[100];
        sprintf(filename, "chunk_size_%d.csv", current_chunk_size);
        FILE *csv_file = fopen(filename, "w");

        // Print to console:
        printf("\n--- Performance Comparison (Chunk Size: %d) ---\n", current_chunk_size);
        printf("Threads\tOpenMP_Time(s)\tGPU_Sim_Time(s)\tOMP_Speedup\tGPU_Speedup\tOMP_Effic\tGPU_Effic\n");
        printf("------------------------------------------------------------------------------------------\n");
        
        fprintf(csv_file, "Threads,OpenMP_Time(s),GPU_Sim_Time(s),OMP_Speedup,GPU_Speedup,OMP_Effic,GPU_Effic,Serial_Time(s)\n");


    // --- Scalability Test ( inside chunk size loop) ---
        for (int p = 1; p <= 64; p *= 2) {
            omp_set_num_threads(p);
            
            // OpenMP cpu fusion:
            double start_openmp = omp_get_wtime();
            fuse_cpu(lidar, radar, fused);
            double end_openmp = omp_get_wtime();
            double openmp_time = end_openmp - start_openmp;
            
            // fake GPU fusion :
            double start_gpu = omp_get_wtime();
            // Pass the current_chunk_size to the function
            fake_gpu(lidar, radar, fused, current_chunk_size);
            double end_gpu = omp_get_wtime();
            double gpu_time = end_gpu - start_gpu;

            // Mterics : 

            // speedup
            double omp_speedup = serial_time / openmp_time;
            double gpu_speedup = serial_time / gpu_time;

         // efficiency
            double omp_efficiency = omp_speedup / (double)p;
            double gpu_efficiency = gpu_speedup / (double)p; 

            // Print to console
            printf("%2d\t%.4f\t\t%.4f\t\t%.2f×\t\t%.2f×\t\t%.2f\t\t%.2f\n",
                   p, openmp_time, gpu_time, omp_speedup, gpu_speedup, omp_efficiency, gpu_efficiency);
                   
            // Print to CSV file
            fprintf(csv_file, "%d,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.4f\n",
                    p, openmp_time, gpu_time, omp_speedup, gpu_speedup, omp_efficiency, gpu_efficiency, serial_time);
        }
        
        fclose(csv_file); // Close the file for this chunk size
        printf("Saved results to %s\n", filename);
    } // --- End of Chunk Size Loop ---


    // printing sample fused data for one random frame
    
    int random_frame = rand() % frames;
    int start_index = random_frame * sectors;
    
    printf("\nSample fused distances for frame %d (showing first 40 sectors):\n", random_frame);
    for (int j = 0; j < 40; j++) { // printing only for the first 40 sectors
        printf("%.2f ", fused[start_index + j]);
        if ((j + 1) % 10 == 0) {
            printf("\n");  
        }
    }
    printf("\n"); 

    // saving all 360 sectors of the sampled frame to CSV:
    
    FILE *sector_csv = fopen("fused_frame_data.csv", "w");
    if (sector_csv) {
        fprintf(sector_csv, "Sector,Fused_Value\n");
        for (int j = 0; j < sectors; j++) {
            fprintf(sector_csv, "%d,%.4f\n", j, fused[start_index + j]);
        }
        fclose(sector_csv);
        printf("Saved all 360 fused sector values for frame %d to fused_frame_data.csv\n", random_frame);
    } else {
        perror("fopen fused_frame_data.csv");
    }


    free(lidar);
    free(radar);
    free(fused);
    return 0;
}
