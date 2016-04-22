package main
 // export GOPATH=$HOME/go
 // go get ./...
 // gonum/plot
 // https://github.com/gonum/plot/wiki/Example-plots
import(
	"fmt"
	"math"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
    "github.com/gonum/plot/plotter"
    "github.com/gonum/plot/plotutil"
    "github.com/gonum/plot/vg"
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

func ToPlotterXYs(denseList []mat64.Dense, w float64, dt float64) plotter.XYs {
    pts := make(plotter.XYs, len(denseList))
    for i := range pts {
		x1 := denseList[i].At(0, 0)
		x2 := denseList[i].At(2, 0)
		n := float64(i)
        pts[i].X = x1*math.Cos(w*n*dt) - x2*math.Sin(w*n*dt) + math.Cos(w*n*dt)
        pts[i].Y = x1*math.Sin(w*n*dt) + x2*math.Cos(w*n*dt) + math.Sin(w*n*dt)
    }
    return pts
}

func main() {

	nbIteration := 500
	epsilon := 0.001
	T := 1.0
	ro := 0.03
	//deltaT := (3*T)/float64(nbIteration)
	deltaT := T/float64(nbIteration)
	fmt.Println(deltaT)
	w := (2*math.Pi)/T

	listeX  := make([]mat64.Dense, nbIteration)
	u := mat64.NewDense( 2, 1, []float64{0, 0})

	for i := 0;  i < nbIteration; i++ {

	    x := mat64.NewDense( 4, 1, []float64{1, 0, 0, 0})
	    A := mat64.NewDense(4, 4, []float64{
	        0,     1,    0,    0,
	        3*w*w, 0,    0,    2*w,
	        0,     0,    0,    1,
	        0,    -2*w,  0,    0,
	    })
	    B := mat64.NewDense(4, 2, []float64{
	        0, 0,
	        1, 0,
	        0, 0,
	        0, 1,
	    })

	    // print the output
		/*
	    fmt.Printf("x = %0.4v\n\n", mat64.Formatted(x, mat64.Prefix("    ")))
	    fmt.Printf("u = %0.4v\n\n", mat64.Formatted(u, mat64.Prefix("    ")))
	    fmt.Printf("A = %0.4v\n\n", mat64.Formatted(A, mat64.Prefix("    ")))
	    fmt.Printf("B = %0.4v\n\n", mat64.Formatted(B, mat64.Prefix("    ")))*/

	    for i := 0; i < nbIteration; i++ {
			x = eulerX(x, u, A, B, deltaT)
	    }
		listeX[i] = *x
		//fmt.Printf("x(%d) = %0.4v\n\n", nbIteration, mat64.Formatted(x, mat64.Prefix("         ")))

		P := x
		for i := nbIteration-1; i >= 0; i-- {
			P = eulerRetroP(P, A, deltaT, int(T))
	    }
		//fmt.Printf("P(%d) = %0.4v\n\n", 0, mat64.Formatted(P, mat64.Prefix("       ")))

		deltaJ := deltaJ(u, B, P, epsilon, int(T))
		//fmt.Printf("deltaJ = %0.4v\n\n", mat64.Formatted(deltaJ, mat64.Prefix("       ")))

		u = unPlus1(u, deltaJ, ro)
		//fmt.Printf("u(%d) = %0.4v\n\n", i, mat64.Formatted(u, mat64.Prefix("         ")))
	}
	//fmt.Printf("u(%d) = %0.4v\n\n", nbIteration, mat64.Formatted(u, mat64.Prefix("       ")))


	/** Affichage **/
	p, err := plot.New()
	if err != nil {
	   panic(err)
	}

	p.Title.Text = "Rendez-vous spatial"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	err = plotutil.AddLinePoints(p, "x", ToPlotterXYs(listeX, w, deltaT))
	if err != nil {
	   panic(err)
	}

	// Save the plot to a PNG file.
	if err := p.Save(7*vg.Inch, 7*vg.Inch, "points2.png"); err != nil {
	   panic(err)
	}
}
