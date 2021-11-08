import iarray as ia
filename="aurelio"
ia.remove_urlpath(filename)

"""
cfg = ia.Config()
cfg2 = ia.Config(urlpath=filename)
print("cfg8")
cfg8 = ia.Config()
print("cfg8",cfg8)
"""



ia.remove_urlpath("bonifasio")
#print("config: ", ia.get_config())
a = ia.linspace([10], start=0, stop=1, urlpath=filename, contiguous=False)
print("linspace: ", a.cfg)

b = ia.linspace([10], start=0, stop=1, urlpath="bonifasio", contiguous=True)
print("linspace b: ", b.cfg)

cfg7 = ia.Config()
print("cfg7", cfg7)

ia.Store()


ia.set_config(contiguous=True)
print("ia.getconfig ", ia.get_config())

cfg8 = ia.Config()
print("cfg8",cfg8)





"""
#print("config: ", ia.get_config())
b = ia.open(filename)
print("open: ", b.cfg)

ia.remove_urlpath(filename)

cfg6 = ia.Config()

c = ia.arange([10], urlpath=filename)
print("arange: ", c.cfg)
ia.remove_urlpath(filename)
cfg5 = ia.Config()

d = ia.empty([10], urlpath=filename)
print("empty: ", d.cfg)
ia.remove_urlpath(filename)

cfg4 = ia.Config()
print("config: ", ia.get_config())
e = ia.arange([10])
print("config e:", e.cfg)

cfg3 = ia.Config()
print("cfg3: ", cfg3)
ia.save(filename, e, contiguous=False)
f = ia.open(filename)
print("config f:", f.cfg)

ia.remove_urlpath(filename)
d = ia.zeros([10], urlpath=filename)
print("zeros: ", d.cfg)
ia.remove_urlpath(filename)

d = ia.ones([10], urlpath=filename)
print("ones: ", d.cfg)
ia.remove_urlpath(filename)


cfg = ia.Config(urlpath=filename)
d = ia.full([10], 3, cfg=cfg)
print("full: ", d.cfg)
ia.remove_urlpath(filename)

"""


