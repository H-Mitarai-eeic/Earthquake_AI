package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os/exec"
	"strconv"
)

type Query struct {
	x     int
	y     int
	depth int
	mag   float64
}

func query_to_string(q Query) map[string]string {
	x := strconv.Itoa(q.x)
	y := strconv.Itoa(q.y)
	depth := strconv.Itoa(q.depth)
	mag := strconv.FormatFloat(q.mag, 'f', -1, 64)
	return map[string]string{"x": x, "y": y, "depth": depth, "mag": mag}
}

func main() {
	const PORT = "8000"
	http.HandleFunc("/", handler)
	log.Fatal(http.ListenAndServe(":"+PORT, nil))
}

func handler(w http.ResponseWriter, r *http.Request) {
	var err error
	w.Header().Set("Access-Control-Allow-Origin", "*")
	fmt.Printf("%s %s %s\n", r.Method, r.URL, r.Proto)

	v := r.URL.Query()
	if v == nil {
		return
	}

	var query Query
	// [MEMO]:
	// 		check whether the queries are legal value
	//		by converting there type
	query.x, err = strconv.Atoi(v["x"][0])
	if err != nil {
		fmt.Println(err)
		return
	}
	query.y, err = strconv.Atoi(v["y"][0])
	if err != nil {
		fmt.Println(err)
		return
	}
	query.depth, err = strconv.Atoi(v["depth"][0])
	if err != nil {
		fmt.Println(err)
		return
	}
	query.mag, err = strconv.ParseFloat(v["mag"][0], 64)
	if err != nil {
		fmt.Println(err)
		return
	}

	// q_st has the query values as string
	q_st := query_to_string(query)
	fmt.Println(q_st)
	err = exec.Command("python3", "python/Hybrid.py", "-x", q_st["x"], "-y", q_st["y"], "-depth", q_st["depth"], "-mag", q_st["mag"]).Run()
	if err != nil {
		fmt.Println(err)
		return
	}

	// data, err := ioutil.ReadFile("python/niigata.csv")
	// data, err := ioutil.ReadFile("python/abn_pre.csv")
	// data, err := ioutil.ReadFile("python/abn_real.csv")
	data, err := ioutil.ReadFile("python/predicted_data.csv")
	if err != nil {
		panic(err)
	}
	fmt.Println("prediction ended")
	fmt.Println()

	// [BE CAREFUL]:
	// 		This header is set for debug.
	// 		If you want to delete this header,
	// 		you have to change the front-end code.
	// header := q_st["x"] + "+" + q_st["y"] + "+" + q_st["depth"] + "+" + q_st["mag"] + ","

	// w.Write([]byte(header + string(data)))
	w.Write([]byte(string(data)))

}
