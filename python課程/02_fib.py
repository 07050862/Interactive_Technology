
def fib(n):
	if n<=2 :
		return 1;
	else:
		return(fib(n-1)+fib(n-2))
x1=10
print("Fib", x1, " without dictionary:", fib(x1))
print('------------------')

fib={0:0,1:1} #Initial Fib Dictionary
def fib_Dict(n):
	keys=fib.keys()
	if n in keys:
		return fib[n]
	else:
		result=fib_Dict(n-1)+fib_Dict(n-2)
		fib[n] = result #Insert a new value in Fib Dictionary
		return(result)
x2=100
print("Fib", x2, " with dictionary:", fib_Dict(x2))