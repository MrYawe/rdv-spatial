package main

import(
	"fmt"
	"math"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/plot"
    "github.com/gonum/plot/plotter"
    "github.com/gonum/plot/plotutil"
    "github.com/gonum/plot/vg"
)

// Constantes
const(
	NB_ITERATION = 500
	EPSILON = 0.001
	T = 1.0
	RO = 0.03
	DELTA_T = T/NB_ITERATION
	W = (2*math.Pi)/T

	PNG_FILE = "./points.png"
)

// f(x) = Ax(t) + Bu(t)
func f(x mat64.Dense, u mat64.Dense, A mat64.Dense, B mat64.Dense) mat64.Dense {
	var Ax, Bu, res mat64.Dense
	Ax.Mul(&A, &x)
	Bu.Mul(&B, &u)
	res.Add(&Ax, &Bu)

	return res
}

// Resolution d'euler explicite à l'itération n+1
func euler(x mat64.Dense, f mat64.Dense, DELTA_T float64) mat64.Dense {
	var res, df mat64.Dense
	df.Scale(DELTA_T, &f)
	res.Add(&x, &df)

	return res
}

// -A^t * p(t)
func g(A mat64.Dense, P mat64.Dense) mat64.Dense {
	var res, AtP mat64.Dense
	AtP.Mul(A.T(), &P)
	res.Scale(-1.0, &AtP)

	return res
}

// Resolution d'euler retrograde à l'itération n+1
func eulerRetrograde(x mat64.Dense, g mat64.Dense, DELTA_T float64) mat64.Dense {
	var res, df mat64.Dense
	df.Scale(DELTA_T, &g)
	res.Sub(&x, &df)

	return res
}

// Calcul du gradiant de J
func deltaJ(u mat64.Dense, B mat64.Dense, P mat64.Dense, EPSILON float64) mat64.Dense {
	var espsilonU, BP, res mat64.Dense
	espsilonU.Scale(EPSILON, &u)
	BP.Mul(B.T(), &P)
	res.Add(&espsilonU, &BP)

	return res
}

// Calcul de x à l'itération n+1
func unPlus1(u mat64.Dense, deltaJ mat64.Dense, ro float64) mat64.Dense {
	var roDeltaJ, res mat64.Dense
	roDeltaJ.Scale(ro, &deltaJ)
	res.Sub(&u, &roDeltaJ)

	return res
}

// Converti le tableau de matrices x en un ensemble de points (on change de repère)
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

	// liste de matrices ou seront stocké les itérations de x
	listeX  := make([]mat64.Dense, NB_ITERATION)

	// Initialisation des matrices
	A := mat64.NewDense(4, 4, []float64{ // note: NewDense renvoie un pointeur de mat64.Dense
		0,     1,    0,    0,
		3*W*W, 0,    0,    2*W,
		0,     0,    0,    1,
		0,    -2*W,  0,    0,
	})
	B := mat64.NewDense(4, 2, []float64{
		0, 0,
		1, 0,
		0, 0,
		0, 1,
	})
	u := mat64.NewDense( 2, 1, []float64{0, 0})

	// Itération principale
	for i := 0;  i < NB_ITERATION; i++ {

		// Initialisation de x
		x := mat64.NewDense( 4, 1, []float64{0.001, 0, 0, 0})

		// Résolution du premier système
	    for i := 0; i < NB_ITERATION; i++ {
			f := f(*x, *u, *A, *B)
			*x = euler(*x, f, DELTA_T)
	    }
		listeX[i] = *x // on ajoute x à la liste
		fmt.Printf("x(%d) = %0.4v\n\n", i, mat64.Formatted(x, mat64.Prefix("         "))) // on affiche x

		// Résolution du second système
		P := *x
		for i := NB_ITERATION-1; i >= 0; i-- {
			g := g(*A, P)
			P = eulerRetrograde(P, g, DELTA_T)
	    }

		// Calcul du gradiant de J
		deltaJ := deltaJ(*u, *B, P, EPSILON)
		// Calcul de u
		*u = unPlus1(*u, deltaJ, RO)
	}


	// Génération du PNG
	p, err := plot.New()
	if err != nil {
	   panic(err)
	}

	p.Title.Text = "Rendez-vous spatial"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	err = plotutil.AddLinePoints(p, "x(n)", ToPlotterXYs(listeX, W, DELTA_T))
	if err != nil {
	   panic(err)
	}

	if err := p.Save(8*vg.Inch, 8*vg.Inch, PNG_FILE); err != nil {
	   panic(err)
	}
	fmt.Printf("File %s created !\n", PNG_FILE)
}
