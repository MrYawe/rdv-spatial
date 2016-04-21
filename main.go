package main
 // export GOPATH=$HOME/go
 // go get ./...
 // gonum/plot
import(
	"fmt"
	"math"
	"github.com/gonum/matrix/mat64"
)

func eulerX(x *mat64.Dense, u *mat64.Dense, A *mat64.Dense, B *mat64.Dense, deltaT float64) *mat64.Dense {
	res := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})

	Ax := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})
	Ax.Mul(A, x)
	dAx := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})
	dAx.Scale(deltaT, Ax)

	Bu := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})
	Bu.Mul(B, u)
	dBu := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})
	dBu.Scale(deltaT, Bu)

	res.Add(dAx, dBu)
	res.Add(res, x)
	return res
}

func eulerRetroP(P *mat64.Dense, A *mat64.Dense, deltaT float64, T int) *mat64.Dense {
	AT := A.T() // transpose

	AtP := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})
	AtP.Mul(AT, P)

	dAtP := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})
	dAtP.Scale(deltaT, AtP)

	res := mat64.NewDense( 4, 1, []float64{0, 0, 0, 0})
	res.Add(P, dAtP)

	return res
}

func deltaJ(u *mat64.Dense, B *mat64.Dense, P *mat64.Dense, epsilon float64, T int) *mat64.Dense {
	espsilonU := mat64.NewDense( 2, 1, []float64{0, 0})
	espsilonU.Scale(epsilon, u)

	BT := B.T() // transpose

	BP := mat64.NewDense( 2, 1, []float64{0, 0})
	BP.Mul(BT, P)

	res := mat64.NewDense( 2, 1, []float64{0, 0})
	res.Add(espsilonU, BP)

	return BP
}

func unPlus1(u *mat64.Dense, deltaJ *mat64.Dense, ro float64) *mat64.Dense {

	roDeltaJ := mat64.NewDense( 2, 1, []float64{0, 0})
	roDeltaJ.Scale(ro, deltaJ)

	res := mat64.NewDense( 2, 1, []float64{0, 0})
	res.Sub(u, roDeltaJ)

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

	u := mat64.NewDense( 2, 1, []float64{0, 0})
	nbIteration := 500

	for i := 0;  i <= nbIteration; i++ {

		epsilon := 0.001
		T := 1.0
		ro := 0.03
		deltaT := (3*T)/float64(nbIteration)
	    var w float64 = (2*math.Pi)/T
	    x := mat64.NewDense( 4, 1, []float64{1, 0, 0, 0})
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
	    /*fmt.Printf("w = %f\n\n", w)
	    fmt.Printf("x = %0.4v\n\n", mat64.Formatted(x, mat64.Prefix("    ")))
	    fmt.Printf("u = %0.4v\n\n", mat64.Formatted(u, mat64.Prefix("    ")))
	    fmt.Printf("A = %0.4v\n\n", mat64.Formatted(A, mat64.Prefix("    ")))
	    fmt.Printf("B = %0.4v\n\n", mat64.Formatted(B, mat64.Prefix("    ")))*/

	    for i := 0; i <= nbIteration; i++ {
			x = eulerX(x, u, A, B, deltaT)
			//fmt.Printf("x(%d) = %0.4v\n\n", i, mat64.Formatted(x, mat64.Prefix("         ")))
	    }
		//fmt.Printf("x(%d) = %0.4v\n\n", nbIteration, mat64.Formatted(x, mat64.Prefix("         ")))

		P := x
		for i := nbIteration-1; i >= 0; i-- {
			P = eulerRetroP(P, A, deltaT, int(T))
			//fmt.Printf("P(%d) = %0.4v\n\n", i, mat64.Formatted(P, mat64.Prefix("         ")))
	    }
		//fmt.Printf("P(%d) = %0.4v\n\n", 0, mat64.Formatted(P, mat64.Prefix("       ")))

		deltaJ := deltaJ(u, B, P, epsilon, int(T))
		//fmt.Printf("deltaJ = %0.4v\n\n", mat64.Formatted(deltaJ, mat64.Prefix("       ")))

		u = unPlus1(u, deltaJ, ro)
		fmt.Printf("u(%d) = %0.4v\n\n", i, mat64.Formatted(u, mat64.Prefix("         ")))
	}
	//fmt.Printf("u(%d) = %0.4v\n\n", nbIteration, mat64.Formatted(u, mat64.Prefix("       ")))





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
