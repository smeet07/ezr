# on my machine, i ran this with:  
#   python3.13 -B extend.py ../moot/optimize/[comp]*/*.csv
import stats
import sys,random
from ezr import the, DATA, csv, dot

def show(lst):
  return print(*[f"{word:6}" for word in lst], sep="\t")

def myfun(train):
  d    = DATA().adds(csv(train))
  x    = len(d.cols.x)
  size = len(d.rows)
  dim  = "small" if x <= 5 else ("med" if x < 12 else "hi")
  size = "small" if size< 500 else ("med" if size<5000 else "hi")
  def guess(N,d):
    some = random.choices(d.rows,k=N)
    some = d.clone().adds(some).chebyshevs().rows
    return some

  somes = []
  if x<6:
    for N in [20,30,40,50]:
      dumb = [guess(N,d) for _ in range(20)]
      dumb = [d.chebyshev(lst[0]) for lst in dumb]
      somes.append(stats.SOME(dumb,txt=f"asIs, {N}"))
      the.Last = N
      # smart = [d.shuffle().activeLearning() for _ in range(20)]
      # smart = [d.chebyshev(lst[0]) for lst in smart]
      # somes.append(stats.SOME(smart,txt=f"smart, {N}"))
      for what,how in scoring_policies:
        for the.Last in [0,20, 30, 40]:
          for the.branch in [False, True]:
            start = time()
            result = []
            runs = 0
            for _ in range(repeats):
              tmp=d.shuffle().activeLearning(score=how)
              runs += len(tmp)
              result += [rnd(d.chebyshev(tmp[0]))]

            pre=f"{what}/b={the.branch}" if the.Last >0 else "rrp"
            tag = f"{pre},{int(runs/repeats)}"
            print(tag, f": {(time() - start) /repeats:.2f} secs")
            somes +=   [stats.SOME(result,    tag)]
      pre=f"{what}/b={the.branch}" if the.Last >0 else "rrp"
      tag = f"{pre},{int(runs/repeats)}"
      somes +=   [stats.SOME(result,    tag)]
    show([train])
    stats.report(somes)
  return [dim, size, x,len(d.cols.y), len(d.rows), train[17:]]

random.seed(the.seed) #  not needed here, but good practice to always take care of seeds
show(["dim", "size","xcols","ycols","rows","file"])
show(["------"] * 6)
[myfun(arg) for arg in sys.argv if arg[-4:] == ".csv"]
