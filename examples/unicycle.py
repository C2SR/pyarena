from pyarena.vehicles.unicycle import Unicycle

# Vehicle parametes
x0 = np.array([0.0,0.0,0])
kwargsUnicycle = {'x0': x0}
vehicle = Unicycle(**kwargsUnicycle)

# Loop
while(1):
    u = np.random.rand(2)
    x = vehicle.run(dt=0.1,u=u)
    print(x)

