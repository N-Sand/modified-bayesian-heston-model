from typing import Dict, Any, Optional

from cmdstanpy import CmdStanModel
import pandas as pd
import arviz as az

from heston_model.utility.helpers import remove_cmdstan_files, parse_stan_dimensions

class StanRunner:
    """
    A generic wrapper which I find more convenient for running Stan models. In this case it is for Heston models.
    """
    
    def __init__(
        self,
        code_path: str = 'heston_model/stan_code/heston.stan',
        cmdstan_outdir : str = './cmdstan'
        ):
        """
        Initialize the Heston model.
        """
        
        self.code_path = code_path
        self.fit = None
        self.idata = None
        self.cmdstan_outdir = cmdstan_outdir
        self.data = None
        
        # compile
        self.model = CmdStanModel(stan_file=code_path)

    def sample(
        self, 
        data: dict, 
        num_samples: int = 1000, 
        chains: int = 4, 
        max_treedepth: int = 11,
        adapt_delta: float = 0.81
        ):
        """
        Sample from the posterior using the provided data.
        """
        # remove existing trace csvs and stdout
        remove_cmdstan_files(self.cmdstan_outdir)
        
        self.data  = data
        
        # sample with progress
        self.fit = self.model.sample(
            data = data, 
            iter_sampling = num_samples,
            chains = chains,
            max_treedepth = max_treedepth,
            output_dir = self.cmdstan_outdir,
            adapt_delta = adapt_delta
            )
        
        return None

    def generate_idata(
        self,
        coords : Dict[str, Any],
        observed_data : Optional[Dict[str, Any]] = None,
        ) -> None:
        
        """
        Generate InferenceData object from the fit.
        """
        
        if not self.fit:
            raise ValueError("Model has not been fit yet.")

        variables, ppcs, dim_symbols = parse_stan_dimensions(self.code_path)
        
        if len(missing_dims := dim_symbols.difference(set(coords.keys()))) != 0:
            raise ValueError(f"Missing dimensions: {missing_dims}, please specify")

        # Create InferenceData object which I find preferable to cmdstanpy/dataframes
        self.idata = az.from_cmdstanpy(
            self.fit,
            observed_data=observed_data,
            posterior_predictive=ppcs,
            coords=coords,
            dims=variables,
        )
        
        return None

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary of the posterior samples.
        """
        if not self.idata:
            raise ValueError("Parse arviz idata before summary.")
        
        return az.summary(self.idata)

    def get_diagnostics(self) -> str:
        """
        Get diagnostics of the posterior samples.
        """
        if not self.fit:
            raise ValueError("Model has not been fit yet.")
        
        return self.fit.diagnose()
    
    def save_trace(
        self,
        dir_path : str = 'trace'
    ) -> None :
        if not self.idata:
            raise ValueError("Create InferenceData object before saving trace.")

        az.to_netcdf(self.idata, dir_path + self.code_path.split('/')[-1].replace('.stan', '.nc'))
        print(f"Trace saved to {dir_path + self.code_path.split('/')[-1].replace('.stan', '.nc')}")
        
