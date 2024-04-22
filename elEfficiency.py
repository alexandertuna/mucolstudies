import ROOT
import glob
ROOT.gROOT.SetBatch()

exec(open("./plotHelper.py").read())

fnames = "plots/electrons/*.root"
#fnames = "plots/electrons_no_el_req/*.root"
files = glob.glob(fnames)
print(f"Found {len(files)} files.")

num = "mcp_el_match"
#num = "trk_el_match"
den = "mcp_el"
plots = ["pt", "eta", "phi"]
#slices = ["250_1000"]
slices = ["0_50", "50_250", "250_1000", "1000_5000"]

for p in plots:

    num_fname = f"{fnames.strip('*.root')}{num}_{p}.root"
    num_file = ROOT.TFile(num_fname, "READ")
    num_can = num_file.Get("can")

    den_fname = f"{fnames.strip('*.root')}{den}_{p}.root"
    den_file = ROOT.TFile(den_fname, "READ")
    den_can = den_file.Get("can")

    hists = {}
    x_range = None
    for s in slices:

        num_h = num_can.GetPrimitive(f"electronGun_pT_{s}_{num}_{p}")#.Rebin(2)
        den_h = den_can.GetPrimitive(f"electronGun_pT_{s}_{den}_{p}")#.Rebin(2)
        eff = ROOT.TEfficiency(num_h, den_h)
        x_range = [num_h.GetXaxis().GetXmin(), num_h.GetXaxis().GetXmax()]
        label = "e p_{T} "+s.split("_")[0]+"-"+s.split("_")[1]
        hists[label] = eff

    try:
        xlabel = variables["obj"][p]["label"]
    except:
        xlabel = variables["evt"][p]["label"]
    print(x_range)
    #plotEfficiencies(hists, "plots/electrons/efftrk_"+p+"_noBIB.png", xlabel=xlabel, ylabel="Efficiency",  xrange=x_range)
    plotEfficiencies(hists, "plots/electrons/eff_"+p+"_noBIB.png", xlabel=xlabel, ylabel="Efficiency",  xrange=x_range)
    #plotEfficiencies(hists, "plots/electrons_no_el_req/eff_"+p+"_noBIB.png", xlabel=xlabel, ylabel="Efficiency",  xrange=x_range)


