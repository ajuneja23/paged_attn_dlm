#pragma once

#include <thrust/device_vector.h>
#include <cuda_runtime.h>



template <typename T> 
struct paged_data_device {
    int num_pages = 0;
    int page_size = 0;
    T** pages = nullptr;
    int last_page_size = 0;
    int agg_seq_len = 0; 

    __host__ __device__ T& operator[] (int idx) {
        int page_idx = idx / page_size; 
        int page_offset = idx % page_size;
        return pages[page_idx][page_offset];
    }
    __host__ __device__ const T& operator[] (int idx) const {
        int page_idx = idx / page_size; 
        int page_offset = idx % page_size;
        return pages[page_idx][page_offset];
    }

};


template <typename T>
struct paged_data_host { //manages paged cache in gpu dram
    int num_pages = 0;
    int page_size = 0;
    thrust::device_vector<T*> d_pages;
    int last_page_size = 0;
    int agg_seq_len = 0;

    explicit paged_data_host (int page_size) : page_size(page_size) {}
    
    void add_data(const T* h_elem) {
        if (num_pages == 0 || last_page_size == page_size) {
            T* new_page;
            cudaMalloc(&new_page, page_size * sizeof(T));
            d_pages.push_back(new_page); 
            num_pages++; 
            cudaMemcpy(new_page, h_elem, sizeof(T), cudaMemcpyHostToDevice);
            last_page_size = 1;
            agg_seq_len++;
        } else {
            cudaMemcpy(d_pages[num_pages - 1] + last_page_size, h_elem, sizeof(T), cudaMemcpyHostToDevice);
            last_page_size++;
            agg_seq_len++;
        }
    }

    ~paged_data_host() {
        for (int i = 0; i < num_pages; i++) {
            cudaFree(d_pages[i]);
        }
    }

    paged_data_device<T> device_data() const {
        paged_data_device<T> dev_data;
        dev_data.num_pages = num_pages;
        dev_data.page_size = page_size;
        dev_data.last_page_size = last_page_size;
        dev_data.agg_seq_len = agg_seq_len;
        dev_data.pages = const_cast<T**>(thrust::raw_pointer_cast(d_pages.data()));
        return dev_data;
    }
};
