test=np.arange(-2,2,0.1)
v1=[]
v2=[]
v3=[]
for i in test:
    i=cp.Constant(i)
    v1.append(pinball(i,0.2).value)
    v2.append(pinball(i,0.5).value)
    v3.append(pinball(i,0.8).value)

plt.plot(test,v1,label="$\\tau=0.2$")
plt.plot(test,v2,label="$\\tau=0.5$")
plt.plot(test,v3,label="$\\tau=0.8$")
plt.xlabel("z")
plt.ylabel("$l_\\tau (z)$")
plt.legend(loc=[1.01, 0.4])
plt.xlim(-2, 2)
plt.ylim(-0.5, 2)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()