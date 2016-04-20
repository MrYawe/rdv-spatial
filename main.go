package main
 // export GOPATH=$HOME/go
import(
  "fmt"
  "math"
  "github.com/gonum/matrix/mat64"
)

func xnplus1(x *mat64.Dense, u *mat64.Dense, A *mat64.Dense, B *mat64.Dense, deltaT float64) *mat64.Dense {
	res := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})

	Ax := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})
	Ax.Mul(A, x)
	Bu := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})
	Bu.Mul(B, u)

	res.Add(Ax, Bu)
	res.Add(res, x)
	return res
}

/*
func test(a mat64.Dense, b mat64.Dense, c mat64.Dense) {
    var m mat64.Dense
    m.Mul(&a, &b)
    fmt.Println(mat64.Formatted(&m))

	var k mat64.Dense
	k.Add(&b, &c)
	fmt.Println(mat64.Formatted(&k))
}*/

func main() {

    // initialize a 3 element float64 slice


    // alternatively, create the matrix y by
    // inserting the data directly as an argument
    //y := mat64.NewDense( 3, 1, []float64{1, 4.57575757, 5})

    // create an empty matrix for the addition
    //z := mat64.NewDense( 3, 1, []float64{0, 0, 0})

    // perform the addition
    //z.Add( x, y )

	deltaT := 0.1
	T := 5480
    w := (2*math.Pi)/5480
    x := mat64.NewDense( 4, 1, []float64{1, 0, 0, 0})
    u := mat64.NewDense( 2, 1, []float64{0, 0})
    A := mat64.NewDense(4, 4, []float64{
        0, 0, 1, 0,
        3*w*w, 0, 0, 2*w,
        0, 0, 0, 1,
        0, -2*w, 0, 0,
    })

    B := mat64.NewDense(4, 2, []float64{
        0, 0,
        1, 0,
        0, 0,
        0, 1,
    })

    // print the output
    //fmt.Printf( "%f %f %f\n", z.At(0,0), z.At(1,0)f, z.At(2,0) Af
    fmt.Printf("w = %f\n\n", w)
    fmt.Printf("x = %0.4v\n\n", mat64.Formatted(x, mat64.Prefix("    ")))
    fmt.Printf("u = %0.4v\n\n", mat64.Formatted(u, mat64.Prefix("    ")))
    fmt.Printf("A = %0.4v\n\n", mat64.Formatted(A, mat64.Prefix("    ")))
    fmt.Printf("B = %0.4v\n\n", mat64.Formatted(B, mat64.Prefix("    ")))

    for i := 0; i <= T; i++ {
		x = xnplus1(x, u, A, B, deltaT)
    }
	fmt.Printf("x(%d) = %0.4v\n\n", T, mat64.Formatted(x, mat64.Prefix("          ")))

	/*
    b := mat64.NewDense(4, 3, []float64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
    })
	c := mat64.NewDense(4, 3, []float64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
    })

    var m mat64.Dense
    m.Mul(A, x)
    fmt.Println(mat64.Formatted(&m))

	var k mat64.Dense
	k.Add(b, c)
	fmt.Println(mat64.Formatted(&k))
	*/

}
