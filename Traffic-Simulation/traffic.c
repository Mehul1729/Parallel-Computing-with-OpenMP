// Multi-lane highway traffic simulation: Serial vs OpenMP vs Fake GPU
#define _POSIX_C_SOURCE 200809L  
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>



// random fucntionw asnot working for a stupid windows error :
static int rand_r(unsigned int *seedp)
{
    *seedp = *seedp * 1103515245 + 12345;
    return (int)((*seedp >> 16) & 0x7FFF);
}

#define VEHICLES 1000

#define VEHICLES 1000
#define LANES 10
#define TIMESTEPS 1000

#define ROAD_LEN 4000.0f      // road length of each lane
#define MAX_V 100.0f           // speed limit
#define ACC 2.0f              // constant acceleration per timestep
#define DT 1.0f               // timestep duration
#define AHEAD_THRESHOLD 20.0f // threshold distance to consider slowing down
#define RANDOM_DECEL_PROB 0.1f
#define RANDOM_DECEL 2.0f     // instantaneous random slow-down

// For fake GPU latency:
#define gpu_latency 20


#define N VEHICLES

typedef struct {
    int id; // Number plate 
    int lane;       // lane index 
    float pos;       //  psoitionon the lane 
    float vel;      // current velocty
} vehicle_t;

static vehicle_t *vehicles;       // current state
static vehicle_t *vehicles_next;  // next state

// for calculating distance ahead:
static inline float distance_ahead(float a, float b) {
    float d = b - a;
    if (d <= 0.0f) d += ROAD_LEN;
    return d;
}


// for randomly distributing vehicles in given lanes :
void init_vehicles(unsigned int seed) {
    srand(seed);

    for (int i = 0; i < N; i++) {
        
        vehicles[i].id = i;
        vehicles[i].lane = rand() % LANES;
        vehicles[i].pos = ((float)rand() / RAND_MAX) * ROAD_LEN; // random position on the lane 
        vehicles[i].vel = ((float)rand() / RAND_MAX) * MAX_V * 0.5f; // random slightly low velocity 

    }
}



// defining the front camera of the cars:
// detects distance of the vehicle ahead and it's ID:



// for a given vehicle having ID = idx and lane = pos+i = vehicles[idx].pos
float front_cam(int idx, int *ahead_idx) {
    float max_distance = ROAD_LEN + 1.0f;

    int ai = -1;
    float pos_i = vehicles[idx].pos;

    int lane_i = vehicles[idx].lane;
    for (int j = 0; j < N; j++) {

        // checking for all cars except the given car in the same lane:
        if (j == idx) continue;
        if (vehicles[j].lane != lane_i) continue;
        float d = distance_ahead(pos_i, vehicles[j].pos);
        if (d < max_distance) {
            max_distance = d;
            ai = j;
        }
    }
    if (ahead_idx) *ahead_idx = ai;
    return max_distance;
}

// single vehicle update logic (given read-only vehicles[] and writes vehicles_next[])
void update_vehicle_step(int i, unsigned int *seedptr) {
    float pos = vehicles[i].pos;
    float vel = vehicles[i].vel;
    int lane = vehicles[i].lane;

    // accelerate up to max
    float desire_vel = vel + ACC * DT;
    if (desire_vel > MAX_V) desire_vel = MAX_V;

    // look for vehicle ahead
    int ahead;
    float gap = front_cam(i, &ahead);

    // if someone too close ahead, reduce desired speed:
    if (gap < AHEAD_THRESHOLD) {
        // desired speed proportional to gap 
        desire_vel = fminf(desire_vel, (gap / AHEAD_THRESHOLD) * MAX_V * 0.5f);
    }

    // random deceleration (driver variability)
    float r = (float)rand_r(seedptr) / RAND_MAX;
    if (r < RANDOM_DECEL_PROB) {
        desire_vel = fmaxf(0.0f, desire_vel - RANDOM_DECEL);
    }

    // move
    float new_pos = pos + desire_vel * DT;
    if (new_pos >= ROAD_LEN) new_pos -= ROAD_LEN; // wrap-around

    vehicles_next[i].id = vehicles[i].id;
    vehicles_next[i].lane = lane; // lane changes not modeled here
    vehicles_next[i].pos = new_pos;
    vehicles_next[i].vel = desire_vel;
}

// ===== Serial baseline version =====
void simulate_serial_collect_csv() {
    unsigned int seed = (unsigned int)time(NULL) ^ 0xA5A5A5A5;
    // Open CSVs to store congestion and density over time (serial baseline)
    FILE *cong = fopen("congestion.csv", "w");
    FILE *dens = fopen("density.csv", "w");
    if (!cong || !dens) {
        perror("fopen congestion/density csv");
        if (cong) fclose(cong);
        if (dens) fclose(dens);
        return;
    }

    // Write headers
    fprintf(cong, "timestep,avg_speed,congestion_fraction\n");
    // density: columns: timestep, lane0, lane1, ..., lane(LANES-1)
    fprintf(dens, "timestep");
    for (int L = 0; L < LANES; L++) fprintf(dens, ",lane_%d_density", L);
    fprintf(dens, "\n");

    for (int t = 0; t < TIMESTEPS; t++) {
        // update all vehicles serially
        for (int i = 0; i < N; i++) {
            update_vehicle_step(i, &seed);
        }
        // swap buffers
        vehicle_t *tmp = vehicles; vehicles = vehicles_next; vehicles_next = tmp;

        // compute metrics
        double total_speed = 0.0;
        int congested_count = 0;
        int lane_counts[LANES] = {0};
        for (int i = 0; i < N; i++) {
            total_speed += vehicles[i].vel;
            if (vehicles[i].vel < (MAX_V * 0.3f)) congested_count++;
            lane_counts[vehicles[i].lane]++;
        }
        double avg_speed = total_speed / (double)N;
        double congestion_fraction = (double)congested_count / (double)N;

        fprintf(cong, "%d,%.6f,%.6f\n", t, avg_speed, congestion_fraction);
        fprintf(dens, "%d", t);
        for (int L = 0; L < LANES; L++) {
            // density as vehicles per 100 units of road length (just a scaled metric)
            double density = (double)lane_counts[L] / (ROAD_LEN / 100.0);
            fprintf(dens, ",%.6f", density);
        }
        fprintf(dens, "\n");
    }

    fclose(cong);
    fclose(dens);
}

// ===== OpenMP CPU version =====
// Parallel updates: each thread computes a subset of vehicles writing into vehicles_next[]
void simulate_openmp(unsigned int *time_seed) {
    unsigned int seed_base;
    #pragma omp parallel private(seed_base)
    {
        seed_base = (unsigned int)time(NULL) + (unsigned int)omp_get_thread_num() * 97;
        unsigned int seed = seed_base;
        for (int t = 0; t < TIMESTEPS; t++) {
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                update_vehicle_step(i, &seed);
            }
            #pragma omp barrier
            #pragma omp single
            {
                vehicle_t *tmp = vehicles; vehicles = vehicles_next; vehicles_next = tmp;
            }
            #pragma omp barrier
        }
    }
}

// ===== Fake GPU version =====
// Using dynamic scheduling with chunk size and small busy-loop for delay
void simulate_fake_gpu(int current_chunk_size) {
    unsigned int seed;
    #pragma omp parallel private(seed)
    {
        seed = (unsigned int)time(NULL) + (unsigned int)omp_get_thread_num()*131;
        for (int t = 0; t < TIMESTEPS; t++) {
            #pragma omp for schedule(dynamic, current_chunk_size)
            for (int i = 0; i < N; i++) {
                // same update logic
                update_vehicle_step(i, &seed);
                // artificial compute to mimic GPU kernel work / memory latency
                for (volatile int k = 0; k < gpu_latency; k++);
            }
            #pragma omp barrier
            #pragma omp single
            {
                vehicle_t *tmp = vehicles; vehicles = vehicles_next; vehicles_next = tmp;
            }
            #pragma omp barrier
        }
    }
}

// ===== MAIN =====
int main() {
    vehicles = (vehicle_t*)malloc(N * sizeof(vehicle_t));
    vehicles_next = (vehicle_t*)malloc(N * sizeof(vehicle_t));
    if (!vehicles || !vehicles_next) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    printf("\n=== TRAFFIC SIMULATION (Serial vs OpenMP vs Fake GPU) ===\n");
    printf("Vehicles: %d | Lanes: %d | Timesteps: %d\n\n", VEHICLES, LANES, TIMESTEPS);

    // Initialize vehicles once (same initial state for all experiments)
    unsigned int seed_init = (unsigned int)time(NULL);
    init_vehicles(seed_init);

    // Make a deep copy for baseline serial (so later experiments start from same initial conditions)
    vehicle_t *vehicles_backup = (vehicle_t*)malloc(N * sizeof(vehicle_t));
    if (!vehicles_backup) { perror("malloc backup"); return 1; }
    memcpy(vehicles_backup, vehicles, N * sizeof(vehicle_t));

    // ------------------ Serial baseline & CSV data ------------------
    printf("Running Serial baseline (also saving congestion.csv and density.csv) ... ");
    fflush(stdout);
    double t0 = omp_get_wtime();
    simulate_serial_collect_csv();
    double t1 = omp_get_wtime();
    double serial_time = t1 - t0;
    printf("Done in %.4f sec\n\n", serial_time);

    // restore initial conditions for parallel experiments
    memcpy(vehicles, vehicles_backup, N * sizeof(vehicle_t));
    memcpy(vehicles_next, vehicles_backup, N * sizeof(vehicle_t)); // good starting next buffer

    // ------------------ Scalability tests ------------------
    // chunk sizes for fake GPU
    int chunk_sizes[] = {4, 8, 16, 32, 64, 128, 256};
    int num_chunk_sizes = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    printf("Threads\tOpenMP_Time(s)\tGPU_Sim_Time(s)\tOMP_Speedup\tGPU_Speedup\tOMP_Effic\tGPU_Effic\n");
    printf("------------------------------------------------------------------------------------------\n");

    for (int c = 0; c < num_chunk_sizes; c++) {
        int current_chunk_size = chunk_sizes[c];
        char filename[128];
        sprintf(filename, "perf_chunk_size_%d.csv", current_chunk_size);
        FILE *csv = fopen(filename, "w");
        if (!csv) { perror("fopen perf csv"); continue; }
        fprintf(csv, "Threads,OpenMP_Time_s,GPU_Sim_Time_s,OMP_Speedup,GPU_Speedup,OMP_Effic,GPU_Effic,Serial_Time_s\n");

        printf("\n--- Performance Comparison (Chunk Size: %d) ---\n", current_chunk_size);

        for (int p = 1; p <= 64; p *= 2) {
            omp_set_num_threads(p);

            // restore initial state before each timed run
            memcpy(vehicles, vehicles_backup, N * sizeof(vehicle_t));
            memcpy(vehicles_next, vehicles_backup, N * sizeof(vehicle_t));

            // OpenMP run
            double start_openmp = omp_get_wtime();
            unsigned int tmp_seed = (unsigned int)time(NULL) ^ 0x12345;
            simulate_openmp(&tmp_seed);
            double end_openmp = omp_get_wtime();
            double openmp_time = end_openmp - start_openmp;

            // restore initial state before GPU sim
            memcpy(vehicles, vehicles_backup, N * sizeof(vehicle_t));
            memcpy(vehicles_next, vehicles_backup, N * sizeof(vehicle_t));

            // fake GPU run
            double start_gpu = omp_get_wtime();
            simulate_fake_gpu(current_chunk_size);
            double end_gpu = omp_get_wtime();
            double gpu_time = end_gpu - start_gpu;

            // metrics
            double omp_speedup = serial_time / openmp_time;
            double gpu_speedup = serial_time / gpu_time;
            double omp_eff = omp_speedup / (double)p;
            double gpu_eff = gpu_speedup / (double)p;

            printf("%2d\t%.4f\t\t%.4f\t\t%.2f×\t\t%.2f×\t\t%.2f\t\t%.2f\n",
                   p, openmp_time, gpu_time, omp_speedup, gpu_speedup, omp_eff, gpu_eff);

            fprintf(csv, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    p, openmp_time, gpu_time, omp_speedup, gpu_speedup, omp_eff, gpu_eff, serial_time);
            fflush(csv);

            // restore initial state for next thread count
            memcpy(vehicles, vehicles_backup, N * sizeof(vehicle_t));
            memcpy(vehicles_next, vehicles_backup, N * sizeof(vehicle_t));
        }

        fclose(csv);
        printf("Saved results to %s\n", filename);
    }

    printf("\nSample: listing first 10 vehicles (id, lane, pos, vel) from serial baseline initial state:\n");
    for (int i = 0; i < 10; i++) {
        printf("%3d: lane=%d pos=%.3f vel=%.3f\n",
               vehicles_backup[i].id, vehicles_backup[i].lane, vehicles_backup[i].pos, vehicles_backup[i].vel);
    }

    free(vehicles_backup);
    free(vehicles);
    free(vehicles_next);


    return 0;
}
