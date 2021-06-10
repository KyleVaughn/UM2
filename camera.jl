fig = Figure()
lscene = LScene(fig[1, 1])
mesh!(whatever)
cameracontrols(lscene.scene).rotationspeed[] = 0.00001f0
