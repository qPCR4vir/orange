import Orange , re
#print "direct: ",re.search(r'H[0-9]+N[0-9]+','22730022.A-whooper swan-Germany-R65-06 H5N1 HP 10-6').group(0)
#HN=re.compile(r'H[0-9]+N[0-9]+')
HN=re.compile(r'N[0-9]+')
name="Type"
out=Orange.data.Table(in_data)
out.domain.add_meta(Orange.feature.Descriptor.new_meta_id(), Orange.feature.String(name))
out.name="Genotyped"
for idx,sample in enumerate(out):
    nh=HN.search(str(sample["probes"]) )
    #print nh.group(0), ' :: ',sample["probes"]
    out[idx][name]= nh.group(0) if nh else '?'#sample["probes"]
out_data=out      
        