ttbb_path=/data1/users/itseyes/23snn/ttbb/Events/
tthbb_path=/data1/users/itseyes/23snn/tthbb/Events/
target_path="./ana_0918/"
mkdir -p $target_path/log

root -l -b -q 'ana_ttbar_df.C("'$ttbb_path'/ttbblj+.root", "'$target_path'/ttbblj+.root")'  &> ./$target_path/log/log00  &
root -l -b -q 'ana_ttbar_df.C("'$ttbb_path'/ttbblj-.root", "'$target_path'/ttbblj-.root")'  &> ./$target_path/log/log01  &

root -l -b -q 'ana_ttbar_df.C("'$tthbb_path'/tthbblj+.root", "'$target_path'/tthbblj+.root")'  &> ./$target_path/log/log50  &
root -l -b -q 'ana_ttbar_df.C("'$tthbb_path'/tthbblj-.root", "'$target_path'/tthbblj-.root")'  &> ./$target_path/log/log51  &
