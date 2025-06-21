#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define TILE_X_SIZE 8
#define TILE_Y_SIZE 16
#define PAGE_SIZE 16 //toks per page




template <typename T1, typename T2>
__device__ void handlePageSDPA(T1* q, T1* k, T1* v, int numKeys T2* output) { 

}

template <typename T1, typename T2>
__device__ void pageHandler(T1* q, T1** k, T1** v, int numPages, int lastPageSize) { 
    int tid=threadIdx.x+blockIdx.x*blockDimx; 
    int warpid=tid/WARP_SIZE;
    int laneid=tid%WARP_SIZE;
// 



template <typename T1, typename T2>
__device__ void allocatePage

template <typename T1, typename T2,int qkv_dim, int pageSize, int num_heads>
class PageHandler {
    private: 
        int numPages; 
        int lastPageSize;
        vector<vector<T1*>> KeyPagePointers; 
        vector<vector<T1*>> ValuePagePointers;
    public:
        PageHandler() {
            numPages=1;
            lastPageSize=0;
            keyPagePointers.add(vector<T1*>());
            valuePagePointers.add(vector<T1*>());
        } 
        void addPage() {
            if (lastPageSize != pageSize) {
                throw std::runtime_error("no need to allocate new page");
            }
            T1* keyPage=nullptr;
            cudaMalloc(&keyPage,pageSize*num_heads*qkv_dim*sizeof(T1));
            T1* valuePage=nullptr;
            cudaMalloc(&valuePage,pageSize*num_heads*qkv_dim*sizeof(T1));
            KeyPagePointers[numPages-1].push_back(keyPage);
            ValuePagePointers[numPages-1].push_back(valuePage);
            lastPageSize=pageSize;
            numPages++;
        }
        void addToken(vector<T1> key, vector<T1> value, int head_idx) {
            if (lastPageSize == pageSize) {
                addPage();
            }
            T1* keyPage=KeyPagePointers[numPages-1][0];

        }



}