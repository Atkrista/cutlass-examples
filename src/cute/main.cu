#include <cutlass/cutlass.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

template <class Shape, class Stride>
void print2D(Layout<Shape, Stride> const &layout)
{
    for (int i = 0; i < size<0>(layout); i++)
    {
        for (int j = 0; j < size<1>(layout); j++)
        {
            printf("%3d ", layout(i, j));
        }
        printf("\n");
    }
}

int main(int argc, char const *argv[])
{
    /* code */
    // auto l = make_layout(make_shape(2, make_shape(1, make_shape(2, 3))));
    const auto col_4x4 = make_layout(make_shape(4, 4));
    const auto row_4x4 = make_layout(make_shape(4, 4), LayoutRight{});
    // print2D(l);
    // print_layout(make_layout(make_shape(3, make_shape(2, 3)), make_stride(2, make_stride(3, 3))));
    float *A = (float *)malloc(sizeof(float) * 16);
    for (unsigned int i = 0; i < 16; i++)
    {
        A[i] = float(i);
    }
    const auto rA = make_tensor(A, row_4x4);
    const auto cA = make_tensor(A, col_4x4);

    printf("Row major: %.2f %.2f %.2f %.2f\n", rA[make_coord(2, 0)], rA[make_coord(2, 1)], rA[make_coord(2, 2)], rA[make_coord(2, 3)]);
    printf("Col major: %.2f %.2f %.2f %.2f\n", cA[make_coord(2, 0)], cA[make_coord(2, 1)], cA[make_coord(2, 2)], cA[make_coord(2, 3)]);
    // print_layout(l);
    // print(rank(l));
    // print(stride(l));
    // print(size(l));
    // print(cosize(l));
    // print(rank<1, 0>(l));
    // print(rank<1, 1>(l));

    return 0;
}
