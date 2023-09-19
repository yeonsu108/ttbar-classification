filepath=./genlevel_0914
target_path="./topreco_0914_2/"
mkdir -p $target_path

root -l -b -q 'topreco_ttbb.C("'$filepath'/ttbblj+_mu.root", "'$target_path'/ttbblj+_mu.root")'  &> ./$target_path/log00  &
root -l -b -q 'topreco_ttbb.C("'$filepath'/ttbblj-_mu.root", "'$target_path'/ttbblj-_mu.root")'  &> ./$target_path/log01  &
root -l -b -q 'topreco_ttbb.C("'$filepath'/ttbblj+_elec.root", "'$target_path'/ttbblj+_elec.root")'  &> ./$target_path/log02  &
root -l -b -q 'topreco_ttbb.C("'$filepath'/ttbblj-_elec.root", "'$target_path'/ttbblj-_elec.root")'  &> ./$target_path/log03  &

root -l -b -q 'topreco_tthbb.C("'$filepath'/tthbblj+_mu.root", "'$target_path'/tthbblj+_mu.root")'  &> ./$target_path/log50  &
root -l -b -q 'topreco_tthbb.C("'$filepath'/tthbblj-_mu.root", "'$target_path'/tthbblj-_mu.root")'  &> ./$target_path/log51  &
root -l -b -q 'topreco_tthbb.C("'$filepath'/tthbblj+_elec.root", "'$target_path'/tthbblj+_elec.root")'  &> ./$target_path/log52  &
root -l -b -q 'topreco_tthbb.C("'$filepath'/tthbblj-_elec.root", "'$target_path'/tthbblj-_elec.root")'  &> ./$target_path/log53  &
