from pathlib import Path
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from open_biomed.data.molecule import Molecule
from open_biomed.utils.exception import MoleculeConstructError


def valid_record(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol) + '\n$$$$\n', mol


@pytest.mark.parametrize('record_count', [1, 2])
def test_no_valid_record_is_construction_error(tmp_path, record_count):
    file = tmp_path / 'invalid.sdf'
    invalid = Chem.MolToMolBlock(
        Chem.MolFromSmiles('CN(C)(C)(C)C', sanitize=False)
    ) + '\n$$$$\n'
    file.write_text(invalid * record_count)
    records = list(Chem.SDMolSupplier(str(file)))
    assert len(records) == record_count
    assert all(m is None for m in records)
    with pytest.raises(MoleculeConstructError, match='No molecule could be parsed') as exc:
        Molecule.from_sdf_file(str(file))
    assert str(file) in str(exc.value)


def test_short_junk_followed_by_valid_record_reports_parse_failure(tmp_path):
    contents, _ = valid_record('CCO')
    file = tmp_path / 'junk_then_valid.sdf'
    file.write_text('bad\nrecord\n$$$$\n' + contents)
    # RDKit can consume the following valid record while rejecting short junk.
    # This records the parser limitation; the loader does not repair the file.
    records = list(Chem.SDMolSupplier(str(file)))
    assert records == [None]
    with pytest.raises(MoleculeConstructError, match='No molecule could be parsed') as exc:
        Molecule.from_sdf_file(str(file))
    assert str(file) in str(exc.value)


def test_single_valid_record_preserves_graph_and_coordinates(tmp_path):
    contents, mol = valid_record('CCO')
    file = tmp_path / 'ethanol.sdf'
    file.write_text(contents)
    result = Molecule.from_sdf_file(str(file))
    assert result.smiles == Chem.MolToSmiles(mol)
    np.testing.assert_allclose(result.conformer, mol.GetConformer().GetPositions(), atol=0.0001)


def test_invalid_then_valid_record_remains_usable(tmp_path):
    contents, mol = valid_record('CCO')
    file = tmp_path / 'mixed.sdf'
    invalid = Chem.MolToMolBlock(Chem.MolFromSmiles('CN(C)(C)(C)C', sanitize=False)) + '\n$$$$\n'
    file.write_text(invalid + contents)
    records = list(Chem.SDMolSupplier(str(file)))
    assert records[0] is None and records[1] is not None
    result = Molecule.from_sdf_file(str(file))
    assert result.smiles == Chem.MolToSmiles(mol)


def test_multiple_valid_records_keep_last_existing_behavior(tmp_path):
    first, _ = valid_record('CCO')
    last, mol = valid_record('c1ccccc1')
    file = tmp_path / 'records.sdf'
    file.write_text(first + last)
    result = Molecule.from_sdf_file(str(file))
    assert result.smiles == Chem.MolToSmiles(mol)


def test_missing_file_preserves_rdkit_oserror(tmp_path):
    with pytest.raises(OSError):
        Molecule.from_sdf_file(str(tmp_path / 'does-not-exist.sdf'))


def test_empty_file_preserves_rdkit_oserror(tmp_path):
    file = tmp_path / 'empty.sdf'
    file.write_text('')
    with pytest.raises(OSError):
        Molecule.from_sdf_file(str(file))
