rm -rf runner
mkdir runner
for i in `ls vasp`
do
python3 RuNNerUC.py vasp_poscar runner -i vasp/$i -o runner/$i
echo $i
done



#energy

rm -f GTSD_Formation.data
j=1
for i in `ls -v runner`
do
deal_file=runner/$i
deal_file2=GTSD_Formation.data
echo begin >> $deal_file2
grep comment $deal_file >> $deal_file2
grep "lattice " $deal_file >> $deal_file2
grep "atom " $deal_file >> $deal_file2
formation=`sed -n "${j}p" formation.data| awk '{print $2}'`
echo -e "energy\t$formation" >> $deal_file2
grep "charge " $deal_file >> $deal_file2
echo -e "comment $i" >> $deal_file2
id=`sed -n "${j}p" formation.data| awk '{print $1}'`
echo -e "comment $id" >> $deal_file2
echo end >> $deal_file2
j=$((j + 1))
echo $i
done