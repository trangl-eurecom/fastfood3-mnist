#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

// The class of FastFoot0Op will yield the transformed version y of x: S * H * G * Pi * H * B
// In this code, I assume that
//  + The number of basis function or random fourier features n is the same with the dimension of input, i.e n = d = 2^k
//  + The randomed diagonal matrix is given from python Tensorflow
REGISTER_OP("FastFood3")
    .Input("in1: float") // x: [Dfeatures, batch_size] tensor
    .Input("in2: float") // diagonal matrix B: [Dfeatures, 1] tensor
    .Input("in3: float") // diagonal matrix P: [Dfeatures, 1] tensor
    .Input("in4: float") // diagonal matrix G: [Dfeatures, 1] tensor
    .Input("in5: float") // diagonal matrix s: [Dfeatures, 1] tensor: each element follows chi distribution
    .Output("out: float") // output: [Dfeatures, batch_size] tensor
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class FastFood3Op : public OpKernel {
    public:
    explicit FastFood3Op(OpKernelConstruction* context) : OpKernel(context) {}
    int batch_size = 0;
    int Dfeatures = 0;
    
    // This function convert const_tensor to usual tensor so that we can easily use tensor in other function
    Tensor convert_const_tensor_to_tensor (const Tensor& tensor)
    {
        TensorShape shape = tensor.shape();
        Tensor t = Tensor(DT_FLOAT, shape);
        auto tensor_flat = tensor.flat<float>();
        auto t_flat = t.flat<float>();
        
        const int N = t_flat.size();
        for (int i = 0; i < N; i++)
        {
            t_flat(i) = tensor_flat(i);
        }
        return t;
    }
    
    int my_pow (int x, int y)
    {
        int p = 1;
        for (int i = 0; i < y; i++)
        {
            p = p * x;
        }
        return p;
    }
    
    // This function will yield the product of Hadamard matrix and tensor
    // This function is based on my algorithm of Hadamard transform
    // Input:
    //  + x_tensor: [Dfeatures, batch_size] tensor
    // Output:
    //  + y2_tensor: [Dfeatures, batch_size] tensor
    Tensor hadamard_multiplication (Tensor& x_tensor)
    {
        Tensor y1_tensor = Tensor(DT_FLOAT, x_tensor.shape());
        Tensor y2_tensor = Tensor(DT_FLOAT, x_tensor.shape());
        auto y1 = y1_tensor.flat<float>();
        auto y2 = y2_tensor.flat<float>();
        
        //Initialize the value of y
        auto x = x_tensor.flat<float>();
        const int N = x.size();
        for (int i = 0; i < N; i++)
        {
            y1(i) = x(i);
            y2(i) = x(i);
        }
        int d; //through level
        int b; //through batch_size
        int D; //through Defeatures
        int c; //coffeficient
        int j; //for update y1
        int nb_element_each_batch;
        int nb_element_each_batch_over_two;
        const int level = (int)(log(this->Dfeatures) / log(2));
        for (d = 0; d < level; d++)
        {
            nb_element_each_batch = my_pow(2, d + 1);
            nb_element_each_batch_over_two = nb_element_each_batch / 2;
            for (b = 0; b < batch_size; b++)
            {
                for (D = 0; D < Dfeatures; D++)
                {
                    c = ((D % nb_element_each_batch) < nb_element_each_batch_over_two);
                    c = 2 * c - 1;
                    //y2(i) = c * y1(i) + y1(i + c * nb_element_each_batch_over_two);
                    y2(D * batch_size + b) = c * y1(D * batch_size + b) + y1((D + c * nb_element_each_batch_over_two) * batch_size + b);
                }
            }
            
            //Update y1
            for (j = 0; j < N; j++)
            {
                y1(j) = y2(j);
            }
        }
        return y2_tensor;
    }
    
    // This function is used to compute the product between a diagonal matrix and a column matrix
    // Input:
    //  + dm_tensor: [Dfeatures, 1]: a diagonal matrix can be represented as an 1-D tensor
    //  + x_tensor: [Dfeatures, batch_size]
    // Output:
    //  + y_tensor: [Dfeatures, batch_size]: a product of dm_tensor and x_tensor
    Tensor diagonal_matrix_mul(Tensor& dm_tensor, Tensor& x_tensor)
    {
        auto dm_tensor_flat = dm_tensor.flat<float>();
        auto x_tensor_flat = x_tensor.flat<float>();
        Tensor y_tensor = Tensor(DT_FLOAT, x_tensor.shape());
        auto y_tensor_flat = y_tensor.flat<float>();
        int bs = 0;
        int D = 0;
        for (D = 0; D < this->Dfeatures; D++)
        {
            for (bs = 0; bs < this->batch_size; bs++)
            {
                y_tensor_flat(D * this->batch_size + bs) = dm_tensor_flat(D) * x_tensor_flat(D * this->batch_size + bs);
            }
        }
        return y_tensor;
    }
    
    // This function will permute the x based on pi
    // Input:
    //  + P_tensor: [Dfeatures, 1] tensor: an arbitrary permutation of [1, Dfeatures]
    //  + x_tensor: [batch_size, Dfeatures] tensor
    // Output:
    //  + y_tensor: [Dfeatures, batch_size] tensor: a permutation of x created by pi
    Tensor permutation(Tensor& P_tensor, Tensor& x_tensor)
    {
        auto P_tensor_flat = P_tensor.flat<float>();
        auto x_tensor_flat = x_tensor.flat<float>();
        Tensor y_tensor = Tensor(DT_FLOAT, x_tensor.shape());
        auto y_tensor_flat = y_tensor.flat<float>();
        
        int bs;
        int D;
        for (bs = 0; bs < this->batch_size; bs++)
        {
            for (D = 0; D < this->Dfeatures; D++)
            {
                y_tensor_flat(D * this->batch_size + bs) = x_tensor_flat(int(P_tensor_flat(D)) * this->batch_size + bs);
            }
        }
        return y_tensor;
    }
    
    // This function is used to make row in x_tensor 'more' independent by dividing each element in x_tensor by frob_norm_G
    // Input:
    //  + G_tensor: [Dfeatures, 1] tensor: a vector of random variables followed normal distribution
    //  + x_tensor: [Dfeatures, batch_size] tensor
    // Output:
    //  + y_tensor: [Dfeatures, batch_size] tensor = frob_norm_G * x_tensor
    Tensor make_row_independent(Tensor& G_tensor, Tensor& x_tensor)
    {
        auto G_tensor_flat = G_tensor.flat<float>();
        auto x_tensor_flat = x_tensor.flat<float>();
        Tensor y_tensor = Tensor(DT_FLOAT, x_tensor.shape());
        auto y_tensor_flat = y_tensor.flat<float>();
        
        // compute the frobenious norm of G
        float frob_norm_G = 0;
        int D = 0;
        for (D = 0 ; D < this->Dfeatures; D++)
        {
            frob_norm_G = frob_norm_G + G_tensor_flat(D) * G_tensor_flat(D);
        }
        frob_norm_G = float(sqrt(double(frob_norm_G)));
        
        // divide frob_norm_G to each element of x_tensor so that the norm of each row in x_tensor is not a constant ==> make rows in x_tensor 'more' independent
        int size = x_tensor_flat.size();
        for (D = 0; D < size; D++)
        {
            y_tensor_flat(D) = x_tensor_flat(D) / frob_norm_G;
        }
        
        return y_tensor;
    }
    
    // This function is used to yield the new transformation of x_tensor corresponding to RBF kernel
    // Input:
    //  + x_tensor: [Dfeatures, batch_size]: input
    //  + B_tensor: [Dfeatures, 1]: random binary scaling array, b[i] is +1 or -1
    //  + P_tensor: [Dfeatures, 1]: random permutation of the set {0,...,d-1}, b[i] = j; i and j is from 1 to d; b[i] != b[j] when i != j
    //  + G_tensor: [Dfeatures, 1]: random Gaussian scaling array, G[i] ~ N(0, 1)
    //  + s_tensor: [Dfeatures, 1]: random variables followed chi distribution, s in paper Fastfood
    // Output:
    //  + y_tensor: [Dfeatures, 1]: the new transformation of x_tensor
    Tensor transform (Tensor& x_tensor, Tensor& B_tensor, Tensor& P_tensor, Tensor& G_tensor, Tensor& S_tensor)
    {
        Tensor B_x = diagonal_matrix_mul(B_tensor, x_tensor);
        Tensor H_B_x = hadamard_multiplication(B_x);
        Tensor P_H_B_x = permutation(P_tensor, H_B_x);
        Tensor G_P_H_B_x = diagonal_matrix_mul(G_tensor, P_H_B_x);
        Tensor H_G_P_H_B_x = hadamard_multiplication(G_P_H_B_x);
        
        // I do not understand how the Gaussian characteristic of x_tensor is remained after multilying with s whose each element follows chi distribution
        Tensor s_H_G_P_H_B_x = diagonal_matrix_mul(S_tensor, H_G_P_H_B_x);
        Tensor S_H_G_P_H_B_x = make_row_independent(G_tensor, s_H_G_P_H_B_x);
        return S_H_G_P_H_B_x;
    }
    
    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        Tensor x_tensor = convert_const_tensor_to_tensor(context->input(0));
        Tensor B_tensor = convert_const_tensor_to_tensor(context->input(1));
        Tensor P_tensor = convert_const_tensor_to_tensor(context->input(2));
        Tensor G_tensor = convert_const_tensor_to_tensor(context->input(3));
        Tensor s_tensor = convert_const_tensor_to_tensor(context->input(4));
        
        // Determine Dfeatures and batch_size
        auto x_tensor_flat = x_tensor.flat<float>();
        const int s = x_tensor_flat.size();
        auto P_tensor_flat =  P_tensor.flat<float>();
        this->Dfeatures = P_tensor_flat.size();
        this->batch_size = s / this->Dfeatures;
        
        // Check the hadamard_multiplication function
        Tensor output_tensor = transform(x_tensor, B_tensor, P_tensor, G_tensor, s_tensor);
        
        // Copy data from output_tensor to output_tensor_pointer
        Tensor* output_tensor_pointer = NULL;
        TensorShape output_shape = output_tensor.shape();
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor_pointer));
        auto output_tensor_pointer_flat = output_tensor_pointer->flat<float>();
        auto output_tensor_flat = output_tensor.flat<float>();
        const int N = output_tensor_flat.size();
        for (int i = 0; i < N; i++)
        {
            output_tensor_pointer_flat(i) = output_tensor_flat(i);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("FastFood3").Device(DEVICE_CPU), FastFood3Op);
