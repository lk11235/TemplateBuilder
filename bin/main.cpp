/**
 *  @file  main.cpp
 *  @brief  
 *
 *
 *  @author  Jean-Baptiste Sauvan <sauvan@llr.in2p3.fr>
 *
 *  @date    12/09/2013
 *
 *  @internal
 *     Created :  12/09/2013
 * Last update :  12/09/2013 11:10:29 PM
 *          by :  JB Sauvan
 *
 * =====================================================================================
 */


/*
#include <string>
#include <iostream>
#include <stdexcept>


#include "TemplateManager.h"
*/

#include "TemplateBuilder.h"
#include "BinTree.h"
#include "GaussKernelSmoother.h"
#include "Smoother1D.h"

#include "TH2F.h"
#include "TH3F.h"
#include "TGraph.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "TemplateParameters.h"

#include "json/json.h"

#include <sstream>
#include <fstream>
#include <stdexcept>

using namespace std;


vector<TH1*> m_widths;

int main() {

    TFile* infile = new TFile("Untagged.root","READ");
    TH3F* histRough = (TH3F*)infile->Get("gg_0");

    //const Json::Value bins = [21,220.,13000.,20,0.,1.,20,-1.,1.]; 

    float bins[] = {21,220.,13000.,20,0.,1.,20,-1.,1.};
    
    vector<unsigned int> nbins;
    vector< pair<double,double> > minmax;
    
    for(unsigned int v=0;v<3;v++)
    {
        nbins.push_back(bins[v*3]);
        minmax.push_back( make_pair(bins[v*3+1],bins[v*3+2]) );
    }

    for(unsigned int axis=0;axis<nbins.size();axis++)
    {
        if(nbins.size()==3)
        {
            m_widths.push_back(new TH3F("Test","Test",nbins[0],minmax[0].first,minmax[0].second,nbins[1],minmax[1].first,minmax[1].second,nbins[2],minmax[2].first,minmax[2].second));
        }
        m_widths.back()->Sumw2();
    }

    GaussKernelSmoother smoother(3);

    std::cout << histRough << std::endl;

    TH3* histSmooth = static_cast<TH3F*>(smoother.smooth(histRough));

    TFile* outfile = new TFile("SmoothedOutput.root","RECREATE");
    histSmooth->Write();
    histRough->Write();
    outfile->Close();
    infile->Close();

    return EXIT_SUCCESS;
    
}

/*
int main(int argc, char** argv)
{
    if(argc!=2)
    {
        std::cerr<<"Usage: buildtemplate.exe parFile.json\n";
        return EXIT_FAILURE;
    }

    std::string parFile(argv[1]);

    TemplateManager manager;
    try
    {
        manager.initialize(parFile);
        manager.loop();
    }catch(std::exception& e)
    {
        std::cerr<<"[ERROR] "<<e.what()<<"\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
*/
