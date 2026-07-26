"""
Computational Electrocatalysis: Water Splitting Project
"""

import sys
from typing import Dict, List, Tuple
from pyscf import gto, scf, mp, cc, dft

# Constants
HARTREE2EV = 27.2114

# --- Utility Functions ---

def build_molecule(atom_str: str, basis: str, spin: int = 0, charge: int = 0) -> gto.Mole:
    """Builds a PySCF molecule object."""
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.spin = spin
    mol.charge = charge
    mol.unit = "Angstrom"
    mol.build()
    return mol

def run_scf(mol: gto.Mole, method: str = "RHF"):
    """Runs a Self-Consistent Field calculation."""
    if method == "RHF":
        mf = scf.RHF(mol)
    elif method == "UHF":
        mf = scf.UHF(mol)
    elif method == "RKS":
        mf = dft.RKS(mol)
        mf.xc = 'pbe'
    elif method == "UKS":
        mf = dft.UKS(mol)
        mf.xc = 'pbe'
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    mf.verbose = 0  # Reduce output noise
    mf.run()
    return mf

# --- Task Execution Functions ---

def run_basis_set_convergence(atoms: Dict[str, str], basis_sets: List[str]) -> List[Tuple[str, float]]:
    """Task 6: Executes the basis set convergence study."""
    results = []
    
    for basis in basis_sets:
        # Build molecules
        mol_h2 = build_molecule(atoms['H2'], basis, spin=0)
        mol_o2 = build_molecule(atoms['O2'], basis, spin=2)
        mol_h2o = build_molecule(atoms['H2O'], basis, spin=0)
        
        # Calculate Energies
        e_h2 = run_scf(mol_h2, "RHF").e_tot
        e_o2 = run_scf(mol_o2, "UHF").e_tot
        e_h2o = run_scf(mol_h2o, "RHF").e_tot
        
        # Calculate Reaction Energy (Eq 2)
        e_r = (2 * e_h2 + e_o2) - (2 * e_h2o)
        results.append((basis, e_r * HARTREE2EV))
        
    return results

def run_high_level_methods(atoms: Dict[str, str], basis: str):
    """Task 8: Executes wave-function method hierarchy (MP2, CCSD)."""
    mol_h2 = build_molecule(atoms['H2'], basis)
    mf_h2 = scf.RHF(mol_h2).run()
    
    e_mp2 = mp.MP2(mf_h2).run().e_tot
    e_ccsd = cc.CCSD(mf_h2).run().e_tot
    
    print(f"\n--- High-Level Methods ({basis}) ---")
    print(f"MP2 Energy: {e_mp2:.6f} Hartrees")
    print(f"CCSD Energy: {e_ccsd:.6f} Hartrees")

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Molecule Definitions
    molecules = {
        'H2': "H 0 0 0; H 0 0 0.74",
        'H2O': "O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586",
        'O2': "O 0 0 0; O 0 0 1.21"
    }

    # 2. Convergence Study
    basis_list = ['sto-3g', '3-21g', '6-31g', 'cc-pvdz', 'cc-pvtz', 'cc-pvqz']
    print("Running Basis Set Convergence Study...")
    convergence_results = run_basis_set_convergence(molecules, basis_list)
    
    for basis, energy in convergence_results:
        print(f"Basis: {basis:<10} | Reaction Energy: {energy:.4f} eV")

    # 3. High-Level Methods
    run_high_level_methods(molecules, basis='cc-pvqz')