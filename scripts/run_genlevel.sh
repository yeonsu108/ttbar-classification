ttbb_path=/data1/users/itseyes/23snn/ttbb/Events/
tthbb_path=/data1/users/itseyes/23snn/tthbb/Events/
target_path="./genlevel/"
mkdir -p $target_path

#root -l -b -q 'genlevel_ttbb.C("ttbblj+", "'$ttbb_path'/ttbblj+.root", "'$target_path'")'  &> ./$target_path/log00  &
#root -l -b -q 'genlevel_ttbb.C("ttbblj-", "'$ttbb_path'/ttbblj-.root", "'$target_path'")'  &> ./$target_path/log01  &

root -l -b -q 'genlevel_tthbb.C("tthbblj+", "'$tthbb_path'/tthbblj+.root", "'$target_path'")'  &> ./$target_path/log50  &
root -l -b -q 'genlevel_tthbb.C("tthbblj-", "'$tthbb_path'/tthbblj-.root", "'$target_path'")'  &> ./$target_path/log51  &
