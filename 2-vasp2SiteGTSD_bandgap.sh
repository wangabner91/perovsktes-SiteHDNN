rm -rf runner
mkdir runner
for i in `ls vasp`
do
python3 RuNNerUC.py vasp_poscar runner -i vasp/$i -o runner/$i
echo $i
done



rm -rf deal-HBF
mkdir deal-HBF
for i in `ls runner`
do
mkdir deal-HBF/$i
cp runner/$i deal-HBF/$i/
sed 's/Br/F/g' deal-HBF/$i/$i > deal-HBF/$i/out_1
sed 's/Cl/F/g' deal-HBF/$i/out_1 > deal-HBF/$i/out_2
sed 's/I/F/g' deal-HBF/$i/out_2 > deal-HBF/$i/out_3
sed 's/Pb/B/g' deal-HBF/$i/out_3 > deal-HBF/$i/out_4
sed 's/Sn/B/g' deal-HBF/$i/out_4 > deal-HBF/$i/out_5
sed 's/Cs/H/g' deal-HBF/$i/out_5 > deal-HBF/$i/out_6
sed 's/Na/H/g' deal-HBF/$i/out_6 > deal-HBF/$i/out_7
sed 's/N/H/g' deal-HBF/$i/out_7 > deal-HBF/$i/out_8
sed 's/C/H/g' deal-HBF/$i/out_8 > deal-HBF/$i/out_9
sed 's/Si/B/g' deal-HBF/$i/out_9 > deal-HBF/$i/out_10
sed 's/Ge/B/g' deal-HBF/$i/out_10 > deal-HBF/$i/out_11
sed 's/Rb/H/g' deal-HBF/$i/out_11 > deal-HBF/$i/out_12
sed 's/K/H/g' deal-HBF/$i/out_12 > deal-HBF/$i/out_13
cp deal-HBF/$i/out_13 deal-HBF/$i/HBF_$i
rm -f deal-HBF/$i/out*
echo $i
done



#bandgap

rm -f SiteGTSD_Bandgap.data
j=1
for i in `ls -v deal-HBF`
do
deal_file=deal-HBF/$i/HBF*
deal_file2=SiteGTSD_Bandgap.data
echo begin >> $deal_file2
grep comment $deal_file >> $deal_file2
grep "lattice " $deal_file >> $deal_file2
grep "atom " $deal_file >> $deal_file2
bandgap=`sed -n "${j}p" bandgap.data| awk '{print $2}'`
echo -e "energy\t$bandgap" >> $deal_file2
grep "charge " $deal_file >> $deal_file2
echo -e "comment $i" >> $deal_file2
id=`sed -n "${j}p" bandgap.data| awk '{print $1}'`
echo -e "comment $id" >> $deal_file2
echo end >> $deal_file2
j=$((j + 1))
echo $i
done