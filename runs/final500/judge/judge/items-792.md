Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 5 items.

=== ITEM id=469a629bb74c1963 ===
Gold sentences and facts:
Sentence 0: Within eukaryotic cells , DNA is organized into long structures called chromosomes . During cell division these chromosomes are duplicated in the process of DNA replication , providing each cell its own complete set of chromosomes . Eukaryotic organisms ( animals , plants , fungi and protists ) store most of their DNA inside the cell nucleus and some of their DNA in organelles , such as mitochondria or chloroplasts . In contrast prokaryotes ( bacteria and archaea ) store their DNA only in the cytoplasm . Within the eukaryotic chromosomes , chromatin proteins such as histones compact and organize DNA . These compact structures guide the interactions between DNA and other proteins , helping control which parts of the DNA are transcribed .
  Fact 0.0: Within eukaryotic cells, DNA is organized into long structures called chromosomes.
  Fact 0.1: During cell division, chromosomes are duplicated in the process of DNA replication.
  Fact 0.2: DNA replication during cell division provides each cell its own complete set of chromosomes.
  Fact 0.3: Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus.
  Fact 0.4: Eukaryotic organisms store some of their DNA in organelles, such as mitochondria or chloroplasts.
  Fact 0.5: Prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in contrast to eukaryotes.
  Fact 0.6: Within eukaryotic chromosomes, chromatin proteins such as histones compact and organize DNA.
  Fact 0.7: The compact structures formed by chromatin proteins guide the interactions between DNA and other proteins.
  Fact 0.8: The compact structures formed by chromatin proteins help control which parts of the DNA are transcribed.
Sentence 1: Genomic DNA is tightly and orderly packed in the process called DNA condensation , to fit the small available volumes of the cell . In eukaryotes , DNA is located in the cell nucleus , with small amounts in mitochondria and chloroplasts . In prokaryotes , the DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid . The genetic information in a genome is held within genes , and the complete set of this information in an organism is called its genotype . A gene is a unit of heredity and is a region of DNA that influences a particular characteristic in an organism . Genes contain an open reading frame that can be transcribed , and regulatory sequences such as promoters and enhancers , which control transcription of the open reading frame .
  Fact 1.0: Genomic DNA is tightly and orderly packed in a process called DNA condensation.
  Fact 1.1: DNA condensation fits genomic DNA into the small available volumes of the cell.
  Fact 1.2: In eukaryotes, DNA is located in the cell nucleus, with small amounts in mitochondria and chloroplasts.
  Fact 1.3: In prokaryotes, DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid.
  Fact 1.4: The genetic information in a genome is held within genes.
  Fact 1.5: The complete set of genetic information in an organism is called its genotype.
  Fact 1.6: A gene is a unit of heredity.
  Fact 1.7: A gene is a region of DNA that influences a particular characteristic in an organism.
  Fact 1.8: Genes contain an open reading frame that can be transcribed.
  Fact 1.9: Genes contain regulatory sequences such as promoters and enhancers.
  Fact 1.10: The regulatory sequences such as promoters and enhancers control transcription of the open reading frame.
Sentence 2: DNA usually occurs as linear chromosomes in eukaryotes , and circular chromosomes in prokaryotes . The set of chromosomes in a cell makes up its genome ; the human genome has approximately 3 billion base pairs of DNA arranged into 46 chromosomes . The information carried by DNA is held in the sequence of pieces of DNA called genes . Transmission of genetic information in genes is achieved via complementary base pairing . For example , in transcription , when a cell uses the information in a gene , the DNA sequence is copied into a complementary RNA sequence through the attraction between the DNA and the correct RNA nucleotides . Usually , this RNA copy is then used to make a matching protein sequence in a process called translation , which depends on the same interaction between RNA nucleotides . In alternative fashion , a cell may simply copy its genetic information in a process called DNA replication . The details of these functions are covered in other articles ; here the focus is on the interactions between DNA and other molecules that mediate the function of the genome .
  Fact 2.0: DNA usually occurs as linear chromosomes in eukaryotes.
  Fact 2.1: DNA usually occurs as circular chromosomes in prokaryotes.
  Fact 2.2: The set of chromosomes in a cell makes up its genome.
  Fact 2.3: The human genome has approximately 3 billion base pairs of DNA.
  Fact 2.4: The human genome's DNA is arranged into 46 chromosomes.
  Fact 2.5: The information carried by DNA is held in the sequence of pieces of DNA called genes.
  Fact 2.6: Transmission of genetic information in genes is achieved via complementary base pairing.
  Fact 2.7: In transcription, when a cell uses the information in a gene, the DNA sequence is copied into a complementary RNA sequence.
  Fact 2.8: The copying in transcription happens through attraction between the DNA and the correct RNA nucleotides.
  Fact 2.9: The RNA copy is usually then used to make a matching protein sequence in a process called translation.
  Fact 2.10: Translation depends on the same interaction between RNA nucleotides.
  Fact 2.11: In an alternative fashion, a cell may simply copy its genetic information in a process called DNA replication.
  Fact 2.12: The details of these functions are covered in other articles.
  Fact 2.13: The focus of this text is on the interactions between DNA and other molecules that mediate the function of the genome.

Retrieved items:
<item 1>
DNA
The DNA sequence may be aligned with other DNA sequences to identify homologous sequences and locate the specific mutations that make them distinct . These
techniques , especially multiple sequence alignment , are used in studying phylogenetic relationships and protein function . Data sets representing entire
genomes ' worth of DNA sequences , such as those produced by the Human Genome Project , are difficult to use without the annotations that identify the locations
of genes and regulatory elements on each chromosome . Regions of DNA sequence that have the characteristic patterns associated with protein - or RNA - coding
genes can be identified by gene finding algorithms , which allow researchers to predict the presence of particular gene products and their possible functions in
an organism even before they have been isolated experimentally . Entire genomes may also be compared , which can shed light on the evolutionary history of
particular organism and permit the examination of complex evolutionary events . DNA nanotechnology The DNA structure at left ( schematic shown ) will self -
assemble into the structure visualized by atomic force microscopy at right . DNA nanotechnology is the field that seeks to design nanoscale structures using the
molecular recognition properties of DNA molecules . Image from Strong , 2004 .
</item 1>

<item 2>
DNA
Levene thought the chain was short and the bases repeated in a fixed order . In 1927 , Nikolai Koltsov proposed that inherited traits would be inherited via a
`` giant hereditary molecule '' made up of `` two mirror strands that would replicate in a semi-conservative fashion using each strand as a template '' . In
1928 , Frederick Griffith in his experiment discovered that traits of the `` smooth '' form of Pneumococcus could be transferred to the `` rough '' form of the
same bacteria by mixing killed `` smooth '' bacteria with the live `` rough '' form . This system provided the first clear suggestion that DNA carries genetic
information . In 1933 , while studying virgin sea urchin eggs , Jean Brachet suggested that DNA is found in cell nucleus and that RNA is present exclusively in
the cytoplasm . At the time , `` yeast nucleic acid '' ( RNA ) was thought to occur only in plants , while `` thymus nucleic acid '' ( DNA ) only in animals .
The latter was thought to be a tetramer , with the function of buffering cellular pH .
</item 2>

<item 3>
DNA
In this way , the base on the old strand dictates which base appears on the new strand , and the cell ends up with a perfect copy of its DNA . Extracellular
nucleic acids Naked extracellular DNA ( eDNA ) , most of it released by cell death , is nearly ubiquitous in the environment . Its concentration in soil may be
as high as 2 μg / L , and its concentration in natural aquatic environments may be as high at 88 μg / L. Various possible functions have been proposed for eDNA
: it may be involved in horizontal gene transfer ; it may provide nutrients ; and it may act as a buffer to recruit or titrate ions or antibiotics .
Extracellular DNA acts as a functional extracellular matrix component in the biofilms of several bacterial species . It may act as a recognition factor to
regulate the attachment and dispersal of specific cell types in the biofilm ; it may contribute to biofilm formation ; and it may contribute to the biofilm 's
physical strength and resistance to biological stress . Cell - free fetal DNA is found in the blood of the mother , and can be sequenced to determine a great
deal of information about the developing fetus .
</item 3>

<item 4>
DNA
This information is replicated as and when the two strands separate . A large part of DNA ( more than 98 % for humans ) is non-coding , meaning that these
sections do not serve as patterns for protein sequences . The two strands of DNA run in opposite directions to each other and are thus antiparallel . Attached
to each sugar is one of four types of nucleobases ( informally , bases ) . It is the sequence of these four nucleobases along the backbone that encodes
biological information . RNA strands are created using DNA strands as a template in a process called transcription . Under the genetic code , these RNA strands
are translated to specify the sequence of amino acids within proteins in a process called translation . Within eukaryotic cells , DNA is organized into long
structures called chromosomes . During cell division these chromosomes are duplicated in the process of DNA replication , providing each cell its own complete
set of chromosomes . Eukaryotic organisms ( animals , plants , fungi and protists ) store most of their DNA inside the cell nucleus and some of their DNA in
organelles , such as mitochondria or chloroplasts .
</item 4>

<item 5>
DNA
Modern biology and biochemistry make intensive use of these techniques in recombinant DNA technology . Recombinant DNA is a man - made DNA sequence that has
been assembled from other DNA sequences . They can be transformed into organisms in the form of plasmids or in the appropriate format , by using a viral vector
. The genetically modified organisms produced can be used to produce products such as recombinant proteins , used in medical research , or be grown in
agriculture . DNA profiling Further information : DNA profiling Forensic scientists can use DNA in blood , semen , skin , saliva or hair found at a crime scene
to identify a matching DNA of an individual , such as a perpetrator . This process is formally termed DNA profiling , but may also be called `` genetic
fingerprinting '' . In DNA profiling , the lengths of variable sections of repetitive DNA , such as short tandem repeats and minisatellites , are compared
between people . This method is usually an extremely reliable technique for identifying a matching DNA . However , identification can be complicated if the
scene is contaminated with DNA from several people .
</item 5>

<item 6>
DNA
In contrast prokaryotes ( bacteria and archaea ) store their DNA only in the cytoplasm . Within the eukaryotic chromosomes , chromatin proteins such as histones
compact and organize DNA . These compact structures guide the interactions between DNA and other proteins , helping control which parts of the DNA are
transcribed . DNA was first isolated by Friedrich Miescher in 1869 . Its molecular structure was first identified by James Watson and Francis Crick at the
Cavendish Laboratory within the University of Cambridge in 1953 , whose model - building efforts were guided by X-ray diffraction data acquired by Raymond
Gosling , who was a post-graduate student of Rosalind Franklin . DNA is used by researchers as a molecular tool to explore physical laws and theories , such as
the ergodic theorem and the theory of elasticity . The unique material properties of DNA have made it an attractive molecule for material scientists and
engineers interested in micro - and nano - fabrication . Among notable advances in this field are DNA origami and DNA - based hybrid materials .
</item 6>

<item 7>
DNA
Image from Strong , 2004 . Further information : DNA nanotechnology DNA nanotechnology uses the unique molecular recognition properties of DNA and other nucleic
acids to create self - assembling branched DNA complexes with useful properties . DNA is thus used as a structural material rather than as a carrier of
biological information . This has led to the creation of two - dimensional periodic lattices ( both tile - based and using the DNA origami method ) and three -
dimensional structures in the shapes of polyhedra . Nanomechanical devices and algorithmic self - assembly have also been demonstrated , and these DNA
structures have been used to template the arrangement of other molecules such as gold nanoparticles and streptavidin proteins . History and anthropology Further
information : Phylogenetics and Genetic genealogy Because DNA collects mutations over time , which are then inherited , it contains historical information , and
, by comparing DNA sequences , geneticists can infer the evolutionary history of organisms , their phylogeny . This field of phylogenetics is a powerful tool in
evolutionary biology . If DNA sequences within a species are compared , population geneticists can learn the history of particular populations .
</item 7>

<item 8>
DNA
Claims for older DNA have been made , most notably a report of the isolation of a viable bacterium from a salt crystal 250 million years old , but these claims
are controversial . Building blocks of DNA ( adenine , guanine , and related organic molecules ) may have been formed extraterrestrially in outer space .
Complex DNA and RNA organic compounds of life , including uracil , cytosine , and thymine , have also been formed in the laboratory under conditions mimicking
those found in outer space , using starting chemicals , such as pyrimidine , found in meteorites . Pyrimidine , like polycyclic aromatic hydrocarbons ( PAHs ) ,
the most carbon - rich chemical found in the universe , may have been formed in red giants or in interstellar cosmic dust and gas clouds . Uses in technology
Genetic engineering Further information : Molecular biology , Nucleic acid methods , and Genetic engineering Methods have been developed to purify DNA from
organisms , such as phenol - chloroform extraction , and to manipulate it in the laboratory , such as restriction digests and the polymerase chain reaction .
Modern biology and biochemistry make intensive use of these techniques in recombinant DNA technology .
</item 8>

<item 9>
DNA
Genes and genomes Further information : Cell nucleus , Chromatin , Chromosome , Gene , and Noncoding DNA Genomic DNA is tightly and orderly packed in the
process called DNA condensation , to fit the small available volumes of the cell . In eukaryotes , DNA is located in the cell nucleus , with small amounts in
mitochondria and chloroplasts . In prokaryotes , the DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid . The genetic
information in a genome is held within genes , and the complete set of this information in an organism is called its genotype . A gene is a unit of heredity and
is a region of DNA that influences a particular characteristic in an organism . Genes contain an open reading frame that can be transcribed , and regulatory
sequences such as promoters and enhancers , which control transcription of the open reading frame . In many species , only a small fraction of the total
sequence of the genome encodes protein . For example , only about 1.5 % of the human genome consists of protein - coding exons , with over 50 % of human DNA
consisting of non-coding repetitive sequences .
</item 9>

<item 10>
DNA
Biological functions Location of eukaryote nuclear DNA within the chromosomes . DNA usually occurs as linear chromosomes in eukaryotes , and circular
chromosomes in prokaryotes . The set of chromosomes in a cell makes up its genome ; the human genome has approximately 3 billion base pairs of DNA arranged into
46 chromosomes . The information carried by DNA is held in the sequence of pieces of DNA called genes . Transmission of genetic information in genes is achieved
via complementary base pairing . For example , in transcription , when a cell uses the information in a gene , the DNA sequence is copied into a complementary
RNA sequence through the attraction between the DNA and the correct RNA nucleotides . Usually , this RNA copy is then used to make a matching protein sequence
in a process called translation , which depends on the same interaction between RNA nucleotides . In alternative fashion , a cell may simply copy its genetic
information in a process called DNA replication . The details of these functions are covered in other articles ; here the focus is on the interactions between
DNA and other molecules that mediate the function of the genome .
</item 10>
=== END ITEM id=469a629bb74c1963 ===

=== ITEM id=26fd51ab91135cea ===
Gold sentences and facts:
Sentence 0: The centerpiece of contemporary Thanksgiving in the United States and Canada is a large meal , generally centered on a large roasted turkey . It is served with a variety of side dishes which vary from traditional dishes such as mashed potatoes , stuffing , and cranberry sauce , to ones that reflect regional or cultural heritage . The majority of the dishes in the traditional American version of Thanksgiving dinner are made from foods native to the New World , as according to tradition the Pilgrims received these foods , or learned how to grow them , from the Native Americans . Thanksgiving dinner is the largest eating event in the United States ; people eat more on Thanksgiving than on any other day of the year .
  Fact 0.0: The centerpiece of contemporary Thanksgiving in the United States and Canada is a large meal.
  Fact 0.1: The large meal is generally centered on a large roasted turkey.
  Fact 0.2: The large meal is served with a variety of side dishes.
  Fact 0.3: The side dishes vary from traditional dishes such as mashed potatoes, stuffing, and cranberry sauce.
  Fact 0.4: The side dishes vary to ones that reflect regional or cultural heritage.
  Fact 0.5: The majority of the dishes in the traditional American version of Thanksgiving dinner are made from foods native to the New World.
  Fact 0.6: According to tradition, the Pilgrims received these foods, or learned how to grow them, from the Native Americans.
  Fact 0.7: Thanksgiving dinner is the largest eating event in the United States.
  Fact 0.8: People eat more on Thanksgiving than on any other day of the year.

Retrieved items:
<item 1>
Thanksgiving dinner
( Pumpkin pie was forbidden because it contained eggs . ) The National Poultry and Egg Board furiously lobbied the President to cease promoting the plan ; it
culminated in a truce at the National Thanksgiving Turkey Presentation shortly before Thanksgiving . Turkey was no longer forbidden , but Eggless Thursdays
remained for the rest of the year , meaning no pumpkin pie was served at the White House dinner that year . Turkey ( edit ) Oven roasted turkey Turkey is the
most common main dish of a Thanksgiving dinner , to the point where Thanksgiving is sometimes colloquially called `` Turkey Day . '' In 2006 , American turkey
growers were expected to raise 270 million turkeys , to be processed into five billion pounds of turkey meat valued at almost $8 billion , with one third of all
turkey consumption occurring in the Thanksgiving - Christmas season , and a per capita consumption of almost 18 pounds ( 8.2 kg ) . The Broad Breasted White
turkey is particularly bred for Thanksgiving dinner and similar large feasts ; its large size ( specimens can grow to over 40 pounds ) and meat content make it
ideal for such situations , although the breed must be artificially bred and suffers from health problems due to its size .
</item 1>

<item 2>
Mayflower
and Little James passengers Plymouth Rock The Mayflower Society Mayflower House Museum Pilgrim Hall Museum Mayflower II Harwich Mayflower Project National
Monument to the Forefathers Pilgrim Monument Thanksgiving History and traditions Canada Samuel de Champlain Martin Frobisher Halifax First Nations United States
Pilgrims Mayflower Plymouth Colony Plymouth , Massachusetts Plymouth Rock Native Americans Samoset Squanto Wampanoag Cornucopia National Thanksgiving
Proclamation Franksgiving National Thanksgiving Turkey Presentation William Penn Sarah Josepha Hale Blackout Wednesday Abraham Lincoln Mourt 's Relation Cuisine
Thanksgiving dinner Turkey Deep - fried Tofurkey Stuffing Gravy Butternut squash Apple cider Hazelnuts Cranberry sauce Carrots Mashed potatoes Sweet potatoes
Acorn squash Yams Corn Green bean casserole Apple pie Pumpkin pie Pecan pie Sweet potato pie Tamale Songs `` Alice 's Restaurant '' `` Bless This House '' ``
Bringing In the Sheaves '' `` Come , Ye Thankful People , Come '' `` For the Beauty of the Earth '' `` Jingle Bells '' `` Let All Things Now Living '' `` Now
Thank We All Our God '' `` Over the River and Through the Wood '' `` Simple Gifts '' `` The Thanksgiving Song '' `` We Gather Together '' `` We Plough the
Fields
</item 2>

<item 3>
Thanksgiving dinner
Massachusetts Plymouth Rock Native Americans Samoset Squanto Wampanoag Cornucopia National Thanksgiving Proclamation Franksgiving National Thanksgiving Turkey
Presentation William Penn Sarah Josepha Hale Blackout Wednesday Abraham Lincoln Mourt 's Relation Cuisine Thanksgiving dinner Turkey Deep - fried Tofurkey
Stuffing Gravy Butternut squash Apple cider Hazelnuts Cranberry sauce Carrots Mashed potatoes Sweet potatoes Acorn squash Yams Corn Green bean casserole Apple
pie Pumpkin pie Pecan pie Sweet potato pie Tamale Songs `` Alice 's Restaurant '' `` Bless This House '' `` Bringing In the Sheaves '' `` Come , Ye Thankful
People , Come '' `` For the Beauty of the Earth '' `` Jingle Bells '' `` Let All Things Now Living '' `` Now Thank We All Our God '' `` Over the River and
Through the Wood '' `` Simple Gifts '' `` The Thanksgiving Song '' `` We Gather Together '' `` We Plough the Fields and Scatter '' Associated events Cultural
Christmas and holiday season Harvest festivals Thanksgiving in film Thanksgiving television specials Winter festivals Lighting of the Macy 's Great Tree World
's Largest Disco Black Friday Parades Novant Health ( Charlotte ) McDonald 's ( Chicago ) America 's ( Detroit ) Oktoberfest ( Kitchener - Waterloo
</item 3>

<item 4>
Thanksgiving dinner
Ham is often served alongside turkey in many non-traditional households . Goose and duck , foods which were traditional European centerpieces of Christmas
dinners before being displaced , are now sometimes served in place of the Thanksgiving turkey . Sometimes , fowl native to the region where the meal is taking
place is used ; for example , an article in Texas Monthly magazine suggested quail as the main dish for a Texan Thanksgiving feast . John Madden , who appeared
on television for the NFL Thanksgiving Day game from 1981 to 2001 , frequently advocated his fondness for the turducken , deboned turkey , duck and chicken
nested inside each other then cooked . In a few areas of the West Coast of the United States , Dungeness crab is common as an alternate main dish , as crab
season starts in early November . Similarly , Thanksgiving falls within deer hunting season in the Northeastern United States , which encourages the use of
venison as a centerpiece . Vegetarians or vegans may have a tofu , wheat gluten or lentil - based substitute ; or stuffed squash . In Alaskan villages , whale
meat is sometimes eaten.
</item 4>

<item 5>
Thanksgiving dinner
Most Thanksgiving turkeys are stuffed with a bread - based mixture and roasted . Sage is the traditional herb added to the stuffing , along with chopped celery
, carrots , and onions . Other ingredients , such as chopped chestnuts or other tree nuts , crumbled sausage or bacon , cranberries , raisins , or apples , may
be added to stuffing . If this mixture is prepared outside the bird , it may be known as dressing . Deep - fried turkey is rising in popularity due to its
shorter preparation time , but carries safety risks . The consumption of turkey on Thanksgiving is so ingrained in American culture that each year since 1947 ,
the National Turkey Federation has presented a live turkey to the President of the United States prior to each Thanksgiving . These turkeys were initially
slaughtered and eaten for the President 's Thanksgiving dinner ; since 1989 , the presented turkeys have typically been given a mock pardon to great fanfare and
sent to a park to live out the rest of their usually short natural lives . Alternatives to turkey ( edit ) Non-traditional foods other than turkey are sometimes
served as the main dish for a Thanksgiving dinner . Ham is often served alongside turkey in many non-traditional households .
</item 5>

<item 6>
Thanksgiving dinner
Copious leftovers are also common following the meal properly . Traditional Thanksgiving foods are sometimes specific to the day of riced potatoes , and
although some of the foods might be seen at any semi-formal meal in the United States , the meal often has something of a ritual or traditional quality . Many
Americans would say it is `` incomplete '' without cranberry sauce , stuffing or dressing , mashed potatoes and gravy , and brussels sprouts . Other commonly
served dishes include winter squash and sweet potatoes , the latter often prepared with sweeteners such as brown sugar , molasses , or marshmallows . Fresh ,
canned , or frozen corn is popular and green beans are frequently served ; in particular , green bean casserole , a product invented in 1955 by the Campbell
Soup Company to promote use of its cream of mushroom soup , has become a Thanksgiving standard . A fresh salad may be included , especially on the West Coast .
Bread rolls or biscuits and cornbread , especially in the South and parts of New England , are served . For dessert , various pies are usually served ,
particularly pumpkin pie , though apple pie , mincemeat pie , sweet potato pie , and pecan pie are often served as well .
</item 6>

<item 7>
Thanksgiving dinner
Thanksgiving dinner - wikipedia Thanksgiving dinner Jump to : navigation , search ( Almost Traditional ) Thanksgiving Dinner The centerpiece of contemporary
Thanksgiving in the United States and Canada is a large meal , generally centered on a large roasted turkey . It is served with a variety of side dishes which
vary from traditional dishes such as mashed potatoes , stuffing , and cranberry sauce , to ones that reflect regional or cultural heritage . The majority of the
dishes in the traditional American version of Thanksgiving dinner are made from foods native to the New World , as according to tradition the Pilgrims received
these foods , or learned how to grow them , from the Native Americans . Thanksgiving dinner is the largest eating event in the United States ; people eat more
on Thanksgiving than on any other day of the year .
</item 7>

<item 8>
Thanksgiving dinner
There are also regional differences as to the stuffing or dressing traditionally served with the turkey . The traditional version has bread cubes , sage , onion
and celery . Southerners generally make their dressing from cornbread , while those in other parts of the country make stuffing from white , wheat or rye bread
as the base . One or several of the following may be added to the dressing / stuffing : oysters , apples , chestnuts , raisins , and sausages or the turkey 's
giblets . Other dishes reflect the region or cultural background of those who have come together for the meal . For example , Sauerkraut ( among those in the
Mid-Atlantic ; especially Baltimore ) is sometimes served . Many African Americans and Southerners serve baked macaroni and cheese and collard greens , along
with chitterlings and sweet potato pie , while some Italian - Americans often have lasagne on the table and Ashkenazi Jews may serve noodle kugel , a sweet
dessert pudding . Other Jewish families may consume foods commonly associated with Hanukkah , such as latkes or a sufganiyah ; the two holidays are usually in
close proximity and on extremely rare occasions overlap .
</item 8>

<item 9>
Thanksgiving dinner
In Alaskan villages , whale meat is sometimes eaten. Irish immigrants have been known to have prime rib of beef as their centerpiece since beef in Ireland was
once a rarity ; families would save up money for this dish to signify newfound prosperity and hope . In the United States , a globalist approach to Thanksgiving
has become common with the impact of immigration . Basic `` Thanksgiving '' ingredients , or the intent of the holiday , can be transformed to a variety of
dishes by using flavors , techniques , and traditions from their own cuisines . Others celebrate the holiday with a variety of dishes particularly when there is
a crowd to be fed , guest 's tastes vary and considering the financial means available . Side dishes ( edit ) A Thanksgiving meal in New England Green bean
casserole Many offerings are typically served alongside the main dish -- so many that , because of the amount of food , the Thanksgiving meal is sometimes
served midday or early afternoon to make time for all the eating , and preparation may begin at dawn or on days prior . Copious leftovers are also common
following the meal properly .
</item 9>

<item 10>
Thanksgiving dinner
McDonald 's ( Chicago ) America 's ( Detroit ) Oktoberfest ( Kitchener - Waterloo ) Hollywood Christmas ( Los Angeles ) Macy 's ( New York City ) 6abc - Dunkin
' Donuts ( Philadelphia ) Celebrate the Season ( Pittsburgh ) America 's Hometown ( Plymouth ) Protests National Day of Mourning Unthanksgiving Day Buy Nothing
Day Sports Football NFL on Thanksgiving Day CFL Thanksgiving Classic English -- Latin rivalry Missouri Turkey Day Game Turkey Day Classic Basketball AdvoCare
Invitational Battle 4 Atlantis Wooden Legacy Maui Invitational Tournament Turkey Trots Atlanta Marathon Berwick Run for the Diamonds Buffalo Turkey Trot Dallas
Turkey Trot Feaster Five Road Race Manchester Road Race Others National Dog Show Pumpkin chucking Turkey bowling Turkey Night Grand Prix Cuisine of the United
States Historical Cuisine of the Thirteen Colonies Thanksgiving dinner Regional List of regional dishes of the United States Allentown , Pennsylvania Appalachia
California Cincinnati Kentucky Hawaii Mid-Atlantic Midwestern Chicago New England New Jersey New Mexico New Orleans New York City Omaha Pacific Northwest
Philadelphia Puerto Rico Southern St.
</item 10>
=== END ITEM id=26fd51ab91135cea ===

=== ITEM id=210f128cab46778f ===
Gold sentences and facts:
Sentence 0: Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 . A sequel , Deadpool 3 , is in development .
  Fact 0.0: Deadpool 2 is scheduled to be released in the United States on May 18, 2018.
  Fact 0.1: A sequel, Deadpool 3, is in development.

Retrieved items:
<item 1>
Deadpool 2
The film is directed by David Leitch from a script by Rhett Reese , Paul Wernick , and Ryan Reynolds , with Reynolds starring in the title role alongside Josh
Brolin , Morena Baccarin , Julian Dennison , Zazie Beetz , T.J. Miller , Brianna Hildebrand , Jack Kesy , and Stefan Kapičić . In Deadpool 2 , Deadpool forms
the team X-Force to protect a young mutant from Cable . Plans for a sequel to Deadpool began before that film 's release , and were confirmed in February 2016 .
Though the original creative team of Reynolds , Reese , Wernick , and director Tim Miller were quickly set to return for the second film , Miller left the
project in October 2016 due to creative differences with Reynolds , and was soon replaced by Leitch . An extensive casting search took place to fill the role of
Cable , with Brolin ultimately cast ; the casting of Beetz as Domino was also noteworthy . Filming took place in British Columbia , Canada , from June to
October 2017 . During filming , stunt woman Joi `` SJ '' Harris died in a motorcycle accident .
</item 1>

<item 2>
Deadpool 2
In January 2018 , the film 's release was moved up to May 18 , 2018 . A month later , Terry Crews was revealed to have a role in the film , the character
Shatterstar was confirmed to be appearing , and the production returned to Vancouver for six days of reshoots under a new working title , Daisy . While some
reports emerged by mid-March claiming that the reshoots were due to poor audience responses during test screenings of the film , and consisted of sweeping
changes , the film was soon confirmed to be testing better than the original did , up to 98 out of 100 over three different tests ( compared to a maximum score
of 91 during tests of the first film ) . Because of this , the reshoots were adding more of the elements that audiences responded positively to , including
additional material featuring Cable and Domino . By the end of the month , two hours - worth of filming had taken place in Los Angeles for a then - secret cameo
appearance , and the film 's first full trailer was released .
</item 2>

<item 3>
Deadpool 2
Officially titling the film Deadpool 2 , the trailer also revealed that Crews was portraying Bedlam ; that Delaney was in the film , portraying Peter ; and that
Bill Skarsgård also had a role . The film 's final trailer , released a month later , confirmed that Lewis Tan would portray Shatterstar , joining Domino ,
Bedlam , and Peter as members of the X-Force in the film . Music Deadpool composer Junkie XL chose not to return for the sequel due to Miller 's departure . In
October 2017 , Tyler Bates was hired to write the score for the film . Release Deadpool 2 is set to be released in the United States on May 18 , 2018 . It was
previously scheduled for release on June 1 of that year . Marketing For the Fox presentation at CineEurope 2017 in June , Reynolds made a video message
featuring himself in costume as Deadpool from the film 's set . The first teaser poster , which pays homage to Norman Rockwell 's 1943 painting Freedom from
Want , was released that November .
</item 3>

<item 4>
Deadpool 2
Retrieved December 16 , 2017 . External links Official website Deadpool 2 on IMDb Deadpool Fabian Nicieza Rob Liefeld Supporting characters Agent X Avengers
Blind Al Bob , Agent of Hydra Sandi Brandenberg Cable Copycat Death Domino Emrys Killebrew Negasonic Teenage Warhead Outlaw Shiklah Siryn Spider - Man Weasel
Wolverine X-Men Antagonists Ajax Black Box Black Swan Black Talon Black Tom Cassidy Bullseye Doctor Bong Dracula Hit - Monkey Flag - Smasher Juggernaut Madcap
Sabretooth Slayback Taskmaster T - Ray Thanos Trapster Typhoid Mary Publications Cable & Deadpool Mrs. Deadpool and the Howling Commandos Wolverine and Deadpool
In other media Deadpool ( video game ) Deadpool ( film ) accolades box office records soundtrack No Good Deed Deadpool 2 Related articles 5 Ronin Gwenpool Mercs
for Money X-Force Deathstroke X-Men in other media Films X-Men X2 X-Men : The Last Stand soundtrack X-Men Origins : Wolverine X-Men : First Class The Wolverine
X-Men : Days of Future Past Deadpool soundtrack X-Men : Apocalypse Logan Deadpool 2 X-Men : Dark Phoenix The New Mutants Development of Gambit Characters
</item 4>

<item 5>
Deadpool 2
As part of a promotion for the film with beverage company Mike 's Harder , Sister Margaret 's School for Wayward Girls -- the fictional bar featured in the
films -- will be emulated in pop - ups at the Alligator Lounge in New York from April 26 -- 28 and the Slipper Clutch in Los Angeles from May 10 -- 12 . The pop
- ups will respectively serve pizza and chimichangas , traditionally Deadpool 's favorite food , as well as Deadpool - inspired Harder drinks . The campaign
also includes the chance to win a trip to the film 's premiere through Harder , and the Los Angeles pop - up will benefit the nonprofit DTLA Film Festival . Box
office projection Initial box office projections for the film in March 2018 expected it to gross $100 million over its opening weekend in the United States and
Canada , with a final domestic total expected to be lower than that of the first film ( $363 million ) at around $242 million . On April 20 , 2018 , both
Fandango and Regal Cinemas announced that Deadpool 2 was the best pre-selling R - rated film in their respective histories .
</item 5>

<item 6>
Deadpool 2
Josh Brolin emerged as a `` surprise contender '' to play Cable in April , ahead of Shannon and Harbour , and was officially cast in the role . Brolin also
portrays the Marvel Comics character Thanos , in the Marvel Cinematic Universe . Leslie Uggams confirmed that she would be reprising her role of Blind Al from
the first film , while Fox gave the sequel a June 1 , 2018 release date . Noting the release date , Leitch wanted to ensure that the film `` was worthy of a
summer tentpole movie , and we knew we were going to be wedged in between some big films '' , specifically wanting to expand the action and make the general
feeling of the film `` bigger '' than the first one . However , he wanted the film to have the same DNA as the original `` in terms of the tone , and the fun
... I love that challenge '' of combining that with the increased scope . The film uses Deadpool 's fourth wall - breaking to reference this release period ,
with Leitch calling these comments `` definitely fresh and timely when they come up . ''
</item 6>

<item 7>
Deadpool 2
During filming , stunt woman Joi `` SJ '' Harris died in a motorcycle accident . Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 .
A sequel , Deadpool 3 , is in development . Contents ( hide ) 1 Premise 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 4
Music 5 Release 5.1 Marketing 6 Box office projection 7 Sequel 8 References 9 External links Premise Deadpool forms a team of mutants called the X-Force to
protect young mutant Russell from the time - traveling soldier Cable . Cast Ryan Reynolds as Wade Wilson / Deadpool : A wisecracking mercenary with accelerated
healing but severe scarring over his body after undergoing an experimental regenerative mutation . He forms the X-Force , a team of mutants . Josh Brolin as
Nathan Summers / Cable : A time travelling cybernetic mutant soldier , `` in many ways the opposite of Deadpool '' .
</item 7>

<item 8>
Deadpool 2
By April 26 , opening weekend industry projections had reached upwards of $150 million . Sequel By November 2016 , with development underway on Deadpool 2 , Fox
was also planning Deadpool 3 , which was said to include the team X-Force . With the confirmation that Leitch would direct Deadpool 2 , it was revealed that Fox
was looking for a separate filmmaker to direct Deadpool 3 . In March 2017 , Reese clarified that though Deadpool 2 sets - up the X-Force team , a future film
focused on the team would be separate from Deadpool 3 , `` so I think we 'll be able to take two paths . ( X-Force ) is where we 're launching something bigger
, but then ( Deadpool 3 is ) where we 're contracting and staying personal and small . '' After the proposed acquisition of 21st Century Fox by Disney was
announced in December 2017 , Disney CEO Bob Iger said that Deadpool would be integrated with the Marvel Cinematic Universe under Disney , and that the company
would be willing to make future R - rated Deadpool films `` as long as we let the audiences know what 's coming '' .
</item 8>

<item 9>
Deadpool 2
Deadpool 2 - wikipedia Deadpool 2 Jump to : navigation , search Deadpool 2 Teaser poster Directed by David Leitch Produced by Simon Kinberg Ryan Reynolds Lauren
Shuler Donner Written by Rhett Reese Paul Wernick Ryan Reynolds Based on Deadpool by Rob Liefeld Fabian Nicieza Starring Ryan Reynolds Josh Brolin Morena
Baccarin Julian Dennison Zazie Beetz T.J. Miller Brianna Hildebrand Jack Kesy Stefan Kapičić Music by Tyler Bates Cinematography Jonathan Sela Edited by Craig
Alpert Elísabet Ronaldsdóttir Production company Marvel Entertainment The Donners ' Company Genre Films Distributed by 20th Century Fox Release date May 18 ,
2018 ( 2018 - 05 - 18 ) ( United States ) Country United States Language English Deadpool 2 is an upcoming American superhero film based on the Marvel Comics
character Deadpool , distributed by 20th Century Fox . It is intended to be the eleventh installment in the X-Men film series , and a sequel to the 2016 film
Deadpool .
</item 9>

<item 10>
Deadpool 2
That 's the biggest mandate going ( into ) the second film ... we have to resist the temptation to make it bigger . -- Producer Simon Kinberg on approaching
Deadpool 2 In June 2016 , Kinberg said that a completed draft from Wernick and Rheese was expected `` soon '' , and that they were looking to begin filming the
sequel at the beginning of 2017 . By August , Kyle Chandler was believed to be in the running to portray Cable . Testing of actresses for Domino had also begun
by October , with the shortlist of actresses under consideration including Lizzy Caplan , Mary Elizabeth Winstead , Sienna Miller , Sofia Boutella , Stephanie
Sigman , Sylvia Hoeks , Mackenzie Davis , Ruby Rose , Eve Hewson , and Kelly Rohrbach . The producers were particularly interested in casting a black or Latina
actress in the role . At the end of October , Miller left the film over `` mutual creative differences '' with Reynolds .
</item 10>
=== END ITEM id=210f128cab46778f ===

=== ITEM id=f439326c78f9d48b ===
Gold sentences and facts:
Sentence 0: No . Name ( birth -- death ) Portrait Party Term of office Electorate served Elections won Ministry Ref Sir Edmund Barton ( 1849 -- 1920 ) Protectionist 1 January 1901 24 September 1903 Hunter , NSW , 1901 -- 1903 ( resigned ) 1901 Barton Alfred Deakin ( 1856 -- 1919 ) Protectionist 24 September 1903 27 April 1904 Ballaarat , Vic , 1901 -- 1913 ( retired ) 1903 1st Deakin Chris Watson ( 1867 -- 1941 ) Labour 27 April 1904 18 August 1904 Bland , NSW , 1901 -- 1906 South Sydney , NSW , 1906 -- 1910 ( retired ) -- Watson George Reid ( 1845 -- 1918 ) Free Trade 18 August 1904 5 July 1905 East Sydney , NSW , 1901 -- 1909 ( resigned ) -- Reid ( 2 ) Alfred Deakin ( 1856 -- 1919 ) Protectionist 5 July 1905 13 November 1908 Ballaarat , Vic , 1901 -- 1913 ( retired ) -- 2nd Deakin 1906 3rd Deakin 5 Andrew Fisher ( 1862 -- 1928 ) Labour 13 November 1908 2 June 1909 Wide Bay , Qld , 1901 -- 1915 ( resigned ) -- 1st Fisher ( 2 ) Alfred Deakin ( 1856 -- 1919 ) Commonwealth Liberal 2 June 1909 29 April 1910 Ballaarat , Vic , 1901 -- 1913 ( retired ) -- 4th Deakin ( 5 ) Andrew Fisher ( 1862 -- 1928 ) Labor 29 April 1910 24 June 1913 Wide Bay , Qld , 1901 -- 1915 ( resigned ) 1910 2nd Fisher 6 Joseph Cook ( 1860 -- 1947 ) Commonwealth Liberal 24 June 1913 17 September 1914 Parramatta , NSW , 1901 -- 1921 ( resigned ) 1913 Cook ( 5 ) Andrew Fisher ( 1862 -- 1928 ) Labor 17 September 1914 27 October 1915 Wide Bay , Qld , 1901 -- 1915 ( resigned ) 1914 3rd Fisher Billy Hughes ( 1862 -- 1952 ) Labor 27 October 1915 14 November 1916 West Sydney , NSW , 1901 -- 1917 Bendigo , Vic , 1917 -- 1922 North Sydney , NSW , 1922 -- 1949 Bradfield , NSW , 1949 -- 1952 ( died ) -- 1st Hughes 7 National Labor 14 November 1916 17 February 1917 -- 2nd Hughes Nationalist 17 February 1917 9 February 1923 -- 3rd Hughes 1917 4th Hughes 1919 5th Hughes 8 Stanley Bruce ( 1883 -- 1967 ) Nationalist ( Coalition ) 9 February 1923 22 October 1929 Flinders , Vic , 1918 -- 1929 ( defeated ) ; 1931 -- 1933 ( resigned ) 1922 1st Bruce 1925 2nd Bruce 1928 3rd Bruce 9 James Scullin ( 1876 -- 1953 ) Labor 22 October 1929 6 January 1932 Corangamite , Vic , 1910 -- 1913 ( defeated ) Yarra , Vic , 1922 -- 1949 ( retired ) 1929 Scullin 10 Joseph Lyons ( 1879 -- 1939 ) United Australia ( Coalition ) 6 January 1932 7 April 1939 † Wilmot , Tas , 1929 -- 1939 ( died ) 1931 1st Lyons 1934 2nd Lyons -- 3rd Lyons 1937 4th Lyons 11 Sir Earle Page ( 1880 -- 1961 ) Country ( Coalition ) 7 April 1939 26 April 1939 Cowper , NSW 1919 -- 1961 ( defeated ) -- Page 12 Robert Menzies ( 1894 -- 1978 ) United Australia ( Coalition ) 26 April 1939 28 August 1941 Kooyong , Vic , 1934 -- 1966 ( resigned ) -- 1st Menzies 2nd Menzies 1940 3rd Menzies 13 Arthur Fadden ( 1894 -- 1973 ) Country ( Coalition ) 28 August 1941 7 October 1941 Darling Downs , Qld 1936 -- 1949 McPherson , Qld 1949 -- 1958 ( retired ) -- Fadden 14 John Curtin ( 1885 -- 1945 ) Labor 7 October 1941 5 July 1945 † Fremantle , WA , 1928 -- 1931 ( defeated ) ; 1934 -- 1945 ( died ) -- 1st Curtin 1943 2nd Curtin 15 Frank Forde ( 1890 -- 1983 ) Labor 6 July 1945 13 July 1945 Capricornia , Qld , 1922 -- 1946 ( defeated ) -- Forde 16 Ben Chifley ( 1885 -- 1951 ) Labor 13 July 1945 19 December 1949 Macquarie , NSW , 1928 -- 1931 ( defeated ) ; 1940 -- 1951 ( died ) -- 1st Chifley 1946 2nd Chifley ( 12 ) Sir Robert Menzies ( 1894 -- 1978 ) Liberal ( Coalition ) 19 December 1949 26 January 1966 Kooyong , Vic , 1934 -- 1966 ( resigned ) 1949 4th Menzies 1951 5th Menzies 1954 6th Menzies 1955 7th Menzies 1958 8th Menzies 1961 9th Menzies 1963 10th Menzies 17 Harold Holt ( 1908 -- 1967 ) Liberal ( Coalition ) 26 January 1966 19 December 1967 † Fawkner , Vic , 1935 -- 1949 Higgins , Vic , 1949 -- 1967 ( disappeared ) -- 1st Holt 1966 2nd Holt 18 John McEwen ( 1900 -- 1980 ) Country ( Coalition ) 19 December 1967 10 January 1968 Echuca , Vic , 1934 -- 1937 Indi , Vic , 1937 -- 1949 Murray , Vic , 1949 -- 1971 ( resigned ) -- McEwen 19 John Gorton ( 1911 -- 2002 ) Liberal ( Coalition ) 10 January 1968 10 March 1971 Senator 1950 -- 1968 ( resigned ) MP for Higgins , Vic , 1968 -- 1975 ( retired ) -- 1st Gorton 1969 2nd Gorton 20 William McMahon ( 1908 -- 1988 ) Liberal ( Coalition ) 10 March 1971 5 December 1972 Lowe , NSW , 1949 -- 1982 ( resigned ) -- McMahon 21 Gough Whitlam ( 1916 -- 2014 ) Labor 5 December 1972 11 November Werriwa , NSW , 1952 -- 1978 ( resigned ) 1972 1st Whitlam -- 2nd Whitlam 3rd Whitlam 22 Malcolm Fraser ( 1930 -- 2015 ) Liberal ( Coalition ) 11 November 11 March Wannon , Vic , 1955 -- 1983 ( resigned ) -- 1st Fraser 2nd Fraser 1977 3rd Fraser 1980 4th Fraser 23 Bob Hawke ( 1929 -- ) Labor 11 March 20 December 1991 Wills , Vic , 1980 -- 1992 ( resigned ) 1st Hawke 1984 2nd Hawke 3rd Hawke 1990 4th Hawke 24 Paul Keating ( 1944 -- ) Labor 20 December 1991 11 March Blaxland , NSW , 1969 -- 1996 ( resigned ) -- 1st Keating 1993 2nd Keating 25 John Howard ( 1939 -- ) Liberal ( Coalition ) 11 March 3 December 2007 Bennelong , NSW , 1974 -- 2007 ( defeated ) 1st Howard 1998 2nd Howard 2001 3rd Howard 4th Howard 26 Kevin Rudd ( 1957 -- ) Labor 3 December 2007 24 June Griffith , Qld , 1998 -- 2013 ( resigned ) 2007 1st Rudd 27 Julia Gillard ( 1961 -- ) Labor 24 June 27 June 2013 Lalor , Vic , 1998 -- 2013 ( retired ) -- 1st Gillard 2nd Gillard ( 26 ) Kevin Rudd ( 1957 -- ) Labor 27 June 2013 18 September 2013 Griffith , Qld , 1998 -- 2013 ( resigned ) -- 2nd Rudd 28 Tony Abbott ( 1957 -- ) Liberal ( Coalition ) 18 September 2013 15 September 2015 Warringah , NSW , since 1994 2013 Abbott 29 Malcolm Turnbull ( 1954 -- ) Liberal ( Coalition ) 15 September 2015 Incumbent Wentworth , NSW , since 2004 -- 1st Turnbull 2016 2nd Turnbull
  Fact 0.0: Sir Edmund Barton (1849 to 1920) was Prime Minister from 1 January 1901 to 24 September 1903.
  Fact 0.1: Edmund Barton was a Protectionist.
  Fact 0.2: Edmund Barton represented Hunter, NSW, from 1901 to 1903 and resigned.
  Fact 0.3: Edmund Barton won the 1901 election.
  Fact 0.4: Edmund Barton led the Barton ministry.
  Fact 0.5: Alfred Deakin (1856 to 1919) was Prime Minister from 24 September 1903 to 27 April 1904.
  Fact 0.6: Alfred Deakin was a Protectionist.
  Fact 0.7: Alfred Deakin represented Ballaarat, Vic, from 1901 to 1913 and retired.
  Fact 0.8: Alfred Deakin won the 1903 election.
  Fact 0.9: Alfred Deakin led the 1st Deakin ministry.
  Fact 0.10: Chris Watson (1867 to 1941) was Prime Minister from 27 April 1904 to 18 August 1904.
  Fact 0.11: Chris Watson was a Labour member.
  Fact 0.12: Chris Watson represented Bland, NSW, from 1901 to 1906.
  Fact 0.13: Chris Watson represented South Sydney, NSW, from 1906 to 1910 and retired.
  Fact 0.14: Chris Watson led the Watson ministry.
  Fact 0.15: George Reid (1845 to 1918) was Prime Minister from 18 August 1904 to 5 July 1905.
  Fact 0.16: George Reid was a Free Trade member.
  Fact 0.17: George Reid represented East Sydney, NSW, from 1901 to 1909 and resigned.
  Fact 0.18: George Reid led the Reid ministry.
  Fact 0.19: Alfred Deakin was Prime Minister again from 5 July 1905 to 13 November 1908.
  Fact 0.20: Alfred Deakin was a Protectionist in this term.
  Fact 0.21: Alfred Deakin led the 2nd Deakin ministry.
  Fact 0.22: Alfred Deakin won the 1906 election.
  Fact 0.23: Alfred Deakin led the 3rd Deakin ministry after the 1906 election.
  Fact 0.24: Andrew Fisher (1862 to 1928) was Prime Minister from 13 November 1908 to 2 June 1909.
  Fact 0.25: Andrew Fisher was a Labour member.
  Fact 0.26: Andrew Fisher represented Wide Bay, Qld, from 1901 to 1915 and resigned.
  Fact 0.27: Andrew Fisher led the 1st Fisher ministry.
  Fact 0.28: Alfred Deakin was Prime Minister from 2 June 1909 to 29 April 1910.
  Fact 0.29: Alfred Deakin was a Commonwealth Liberal in this term.
  Fact 0.30: Alfred Deakin led the 4th Deakin ministry.
  Fact 0.31: Andrew Fisher was Prime Minister from 29 April 1910 to 24 June 1913.
  Fact 0.32: Andrew Fisher was a Labor member in this term.
  Fact 0.33: Andrew Fisher won the 1910 election.
  Fact 0.34: Andrew Fisher led the 2nd Fisher ministry.
  Fact 0.35: Joseph Cook (1860 to 1947) was Prime Minister from 24 June 1913 to 17 September 1914.
  Fact 0.36: Joseph Cook was a Commonwealth Liberal.
  Fact 0.37: Joseph Cook represented Parramatta, NSW, from 1901 to 1921 and resigned.
  Fact 0.38: Joseph Cook won the 1913 election.
  Fact 0.39: Joseph Cook led the Cook ministry.
  Fact 0.40: Andrew Fisher was Prime Minister from 17 September 1914 to 27 October 1915.
  Fact 0.41: Andrew Fisher was a Labor member in this term.
  Fact 0.42: Andrew Fisher won the 1914 election.
  Fact 0.43: Andrew Fisher led the 3rd Fisher ministry.
  Fact 0.44: Billy Hughes (1862 to 1952) was Prime Minister from 27 October 1915 to 14 November 1916.
  Fact 0.45: Billy Hughes was a Labor member.
  Fact 0.46: Billy Hughes represented West Sydney, NSW, from 1901 to 1917.
  Fact 0.47: Billy Hughes represented Bendigo, Vic, from 1917 to 1922.
  Fact 0.48: Billy Hughes represented North Sydney, NSW, from 1922 to 1949.
  Fact 0.49: Billy Hughes represented Bradfield, NSW, from 1949 to 1952, ending with his death.
  Fact 0.50: Billy Hughes led the 1st Hughes ministry.
  Fact 0.51: Billy Hughes was Prime Minister from 14 November 1916 to 17 February 1917 as a National Labor member.
  Fact 0.52: Billy Hughes led the 2nd Hughes ministry.
  Fact 0.53: Billy Hughes was Prime Minister from 17 February 1917 to 9 February 1923 as a Nationalist.
  Fact 0.54: Billy Hughes led the 3rd Hughes ministry.
  Fact 0.55: Billy Hughes won the 1917 election and led the 4th Hughes ministry.
  Fact 0.56: Billy Hughes won the 1919 election and led the 5th Hughes ministry.
  Fact 0.57: Stanley Bruce (1883 to 1967) was Prime Minister from 9 February 1923 to 22 October 1929.
  Fact 0.58: Stanley Bruce was a Nationalist (Coalition) member.
  Fact 0.59: Stanley Bruce represented Flinders, Vic, from 1918 to 1929 and was defeated.
  Fact 0.60: Stanley Bruce represented Flinders, Vic, from 1931 to 1933 and resigned.
  Fact 0.61: Stanley Bruce won the 1922 election and led the 1st Bruce ministry.
  Fact 0.62: Stanley Bruce won the 1925 election and led the 2nd Bruce ministry.
  Fact 0.63: Stanley Bruce won the 1928 election and led the 3rd Bruce ministry.
  Fact 0.64: James Scullin (1876 to 1953) was Prime Minister from 22 October 1929 to 6 January 1932.
  Fact 0.65: James Scullin was a Labor member.
  Fact 0.66: James Scullin represented Corangamite, Vic, from 1910 to 1913 and was defeated.
  Fact 0.67: James Scullin represented Yarra, Vic, from 1922 to 1949 and retired.
  Fact 0.68: James Scullin won the 1929 election and led the Scullin ministry.
  Fact 0.69: Joseph Lyons (1879 to 1939) was Prime Minister from 6 January 1932 to 7 April 1939.
  Fact 0.70: Joseph Lyons was a United Australia (Coalition) member.
  Fact 0.71: Joseph Lyons represented Wilmot, Tas, from 1929 to 1939 and died in office.
  Fact 0.72: Joseph Lyons won the 1931 election and led the 1st Lyons ministry.
  Fact 0.73: Joseph Lyons won the 1934 election and led the 2nd Lyons ministry.
  Fact 0.74: Joseph Lyons led the 3rd Lyons ministry.
  Fact 0.75: Joseph Lyons won the 1937 election and led the 4th Lyons ministry.
  Fact 0.76: Sir Earle Page (1880 to 1961) was Prime Minister from 7 April 1939 to 26 April 1939.
  Fact 0.77: Earle Page was a Country (Coalition) member.
  Fact 0.78: Earle Page represented Cowper, NSW, from 1919 to 1961 and was defeated.
  Fact 0.79: Earle Page led the Page ministry.
  Fact 0.80: Robert Menzies (1894 to 1978) was Prime Minister from 26 April 1939 to 28 August 1941.
  Fact 0.81: Robert Menzies was a United Australia (Coalition) member.
  Fact 0.82: Robert Menzies represented Kooyong, Vic, from 1934 to 1966 and resigned.
  Fact 0.83: Robert Menzies led the 1st Menzies ministry.
  Fact 0.84: Robert Menzies led the 2nd Menzies ministry.
  Fact 0.85: Robert Menzies won the 1940 election and led the 3rd Menzies ministry.
  Fact 0.86: Arthur Fadden (1894 to 1973) was Prime Minister from 28 August 1941 to 7 October 1941.
  Fact 0.87: Arthur Fadden was a Country (Coalition) member.
  Fact 0.88: Arthur Fadden represented Darling Downs, Qld, from 1936 to 1949.
  Fact 0.89: Arthur Fadden represented McPherson, Qld, from 1949 to 1958 and retired.
  Fact 0.90: Arthur Fadden led the Fadden ministry.
  Fact 0.91: John Curtin (1885 to 1945) was Prime Minister from 7 October 1941 to 5 July 1945.
  Fact 0.92: John Curtin was a Labor member.
  Fact 0.93: John Curtin represented Fremantle, WA, from 1928 to 1931 and was defeated.
  Fact 0.94: John Curtin represented Fremantle, WA, from 1934 to 1945 and died in office.
  Fact 0.95: John Curtin led the 1st Curtin ministry.
  Fact 0.96: John Curtin won the 1943 election and led the 2nd Curtin ministry.
  Fact 0.97: Frank Forde (1890 to 1983) was Prime Minister from 6 July 1945 to 13 July 1945.
  Fact 0.98: Frank Forde was a Labor member.
  Fact 0.99: Frank Forde represented Capricornia, Qld, from 1922 to 1946 and was defeated.
  Fact 0.100: Frank Forde led the Forde ministry.
  Fact 0.101: Ben Chifley (1885 to 1951) was Prime Minister from 13 July 1945 to 19 December 1949.
  Fact 0.102: Ben Chifley was a Labor member.
  Fact 0.103: Ben Chifley represented Macquarie, NSW, from 1928 to 1931 and was defeated.
  Fact 0.104: Ben Chifley represented Macquarie, NSW, from 1940 to 1951 and died in office.
  Fact 0.105: Ben Chifley led the 1st Chifley ministry.
  Fact 0.106: Ben Chifley won the 1946 election and led the 2nd Chifley ministry.
  Fact 0.107: Sir Robert Menzies (1894 to 1978) was Prime Minister from 19 December 1949 to 26 January 1966.
  Fact 0.108: Robert Menzies was a Liberal (Coalition) member in this term.
  Fact 0.109: Robert Menzies represented Kooyong, Vic, from 1934 to 1966 and resigned.
  Fact 0.110: Robert Menzies won elections in 1949, 1951, 1954, 1955, 1958, 1961 and 1963.
  Fact 0.111: Robert Menzies led the 4th through 10th Menzies ministries.
  Fact 0.112: Harold Holt (1908 to 1967) was Prime Minister from 26 January 1966 to 19 December 1967 and disappeared in office.
  Fact 0.113: Harold Holt was a Liberal (Coalition) member.
  Fact 0.114: Harold Holt represented Fawkner, Vic, from 1935 to 1949.
  Fact 0.115: Harold Holt represented Higgins, Vic, from 1949 to 1967, ending when he disappeared.
  Fact 0.116: Harold Holt led the 1st Holt ministry.
  Fact 0.117: Harold Holt won the 1966 election and led the 2nd Holt ministry.
  Fact 0.118: John McEwen (1900 to 1980) was Prime Minister from 19 December 1967 to 10 January 1968.
  Fact 0.119: John McEwen was a Country (Coalition) member.
  Fact 0.120: John McEwen represented Echuca, Vic, from 1934 to 1937.
  Fact 0.121: John McEwen represented Indi, Vic, from 1937 to 1949.
  Fact 0.122: John McEwen represented Murray, Vic, from 1949 to 1971 and resigned.
  Fact 0.123: John McEwen led the McEwen ministry.
  Fact 0.124: John Gorton (1911 to 2002) was Prime Minister from 10 January 1968 to 10 March 1971.
  Fact 0.125: John Gorton was a Liberal (Coalition) member.
  Fact 0.126: John Gorton was a Senator from 1950 to 1968 and resigned.
  Fact 0.127: John Gorton was MP for Higgins, Vic, from 1968 to 1975 and retired.
  Fact 0.128: John Gorton led the 1st Gorton ministry.
  Fact 0.129: John Gorton won the 1969 election and led the 2nd Gorton ministry.
  Fact 0.130: William McMahon (1908 to 1988) was Prime Minister from 10 March 1971 to 5 December 1972.
  Fact 0.131: William McMahon was a Liberal (Coalition) member.
  Fact 0.132: William McMahon represented Lowe, NSW, from 1949 to 1982 and resigned.
  Fact 0.133: William McMahon led the McMahon ministry.
  Fact 0.134: Gough Whitlam (1916 to 2014) was Prime Minister from 5 December 1972 to 11 November.
  Fact 0.135: Gough Whitlam was a Labor member.
  Fact 0.136: Gough Whitlam represented Werriwa, NSW, from 1952 to 1978 and resigned.
  Fact 0.137: Gough Whitlam won the 1972 election and led the 1st Whitlam ministry.
  Fact 0.138: Gough Whitlam led the 2nd Whitlam ministry.
  Fact 0.139: Gough Whitlam led the 3rd Whitlam ministry.
  Fact 0.140: Malcolm Fraser (1930 to 2015) was Prime Minister from 11 November to 11 March.
  Fact 0.141: Malcolm Fraser was a Liberal (Coalition) member.
  Fact 0.142: Malcolm Fraser represented Wannon, Vic, from 1955 to 1983 and resigned.
  Fact 0.143: Malcolm Fraser led the 1st Fraser ministry.
  Fact 0.144: Malcolm Fraser led the 2nd Fraser ministry.
  Fact 0.145: Malcolm Fraser won the 1977 election and led the 3rd Fraser ministry.
  Fact 0.146: Malcolm Fraser won the 1980 election and led the 4th Fraser ministry.
  Fact 0.147: Bob Hawke (born 1929) was Prime Minister from 11 March to 20 December 1991.
  Fact 0.148: Bob Hawke was a Labor member.
  Fact 0.149: Bob Hawke represented Wills, Vic, from 1980 to 1992 and resigned.
  Fact 0.150: Bob Hawke led the 1st Hawke ministry.
  Fact 0.151: Bob Hawke won the 1984 election and led the 2nd Hawke ministry.
  Fact 0.152: Bob Hawke led the 3rd Hawke ministry.
  Fact 0.153: Bob Hawke won the 1990 election and led the 4th Hawke ministry.
  Fact 0.154: Paul Keating (born 1944) was Prime Minister from 20 December 1991 to 11 March.
  Fact 0.155: Paul Keating was a Labor member.
  Fact 0.156: Paul Keating represented Blaxland, NSW, from 1969 to 1996 and resigned.
  Fact 0.157: Paul Keating led the 1st Keating ministry.
  Fact 0.158: Paul Keating won the 1993 election and led the 2nd Keating ministry.
  Fact 0.159: John Howard (born 1939) was Prime Minister from 11 March to 3 December 2007.
  Fact 0.160: John Howard was a Liberal (Coalition) member.
  Fact 0.161: John Howard represented Bennelong, NSW, from 1974 to 2007 and was defeated.
  Fact 0.162: John Howard led the 1st Howard ministry.
  Fact 0.163: John Howard won the 1998 election and led the 2nd Howard ministry.
  Fact 0.164: John Howard won the 2001 election and led the 3rd Howard ministry.
  Fact 0.165: John Howard led the 4th Howard ministry.
  Fact 0.166: Kevin Rudd (born 1957) was Prime Minister from 3 December 2007 to 24 June.
  Fact 0.167: Kevin Rudd was a Labor member.
  Fact 0.168: Kevin Rudd represented Griffith, Qld, from 1998 to 2013 and resigned.
  Fact 0.169: Kevin Rudd won the 2007 election and led the 1st Rudd ministry.
  Fact 0.170: Julia Gillard (born 1961) was Prime Minister from 24 June to 27 June 2013.
  Fact 0.171: Julia Gillard was a Labor member.
  Fact 0.172: Julia Gillard represented Lalor, Vic, from 1998 to 2013 and retired.
  Fact 0.173: Julia Gillard led the 1st Gillard ministry.
  Fact 0.174: Julia Gillard led the 2nd Gillard ministry.
  Fact 0.175: Kevin Rudd was Prime Minister again from 27 June 2013 to 18 September 2013.
  Fact 0.176: Kevin Rudd was a Labor member.
  Fact 0.177: Kevin Rudd represented Griffith, Qld, from 1998 to 2013 and resigned.
  Fact 0.178: Kevin Rudd led the 2nd Rudd ministry.
  Fact 0.179: Tony Abbott (born 1957) was Prime Minister from 18 September 2013 to 15 September 2015.
  Fact 0.180: Tony Abbott was a Liberal (Coalition) member.
  Fact 0.181: Tony Abbott has represented Warringah, NSW, since 1994.
  Fact 0.182: Tony Abbott won the 2013 election and led the Abbott ministry.
  Fact 0.183: Malcolm Turnbull (born 1954) has been Prime Minister since 15 September 2015.
  Fact 0.184: Malcolm Turnbull is a Liberal (Coalition) member.
  Fact 0.185: Malcolm Turnbull has represented Wentworth, NSW, since 2004.
  Fact 0.186: Malcolm Turnbull led the 1st Turnbull ministry.
  Fact 0.187: Malcolm Turnbull won the 2016 election and led the 2nd Turnbull ministry.
  Fact 0.188: Malcolm Turnbull is still in office.
Sentence 1: Federation occurred on 1 January 1901 , but elections for the first parliament were not scheduled until late March . In the interim , an unelected caretaker government was necessary . In what is now known as the Hopetoun Blunder , the governor - general , Lord Hopetoun , invited Sir William Lyne , the premier of the most populous state , New South Wales , to form a government . Lyne was unable to do so and returned his commission in favour of Edmund Barton , who became the first Prime Minister and led the inaugural government into and beyond the election .
  Fact 1.0: Federation occurred on 1 January 1901.
  Fact 1.1: Elections for the first parliament were not scheduled until late March.
  Fact 1.2: In the interim, an unelected caretaker government was necessary.
  Fact 1.3: The event is now known as the Hopetoun Blunder.
  Fact 1.4: The governor-general, Lord Hopetoun, invited Sir William Lyne to form a government.
  Fact 1.5: Sir William Lyne was the premier of New South Wales.
  Fact 1.6: New South Wales was the most populous state.
  Fact 1.7: Sir William Lyne was unable to form a government.
  Fact 1.8: Sir William Lyne returned his commission in favour of Edmund Barton.
  Fact 1.9: Edmund Barton became the first Prime Minister.
  Fact 1.10: Edmund Barton led the inaugural government into the election.
  Fact 1.11: Edmund Barton led the inaugural government beyond the election.

Retrieved items:
<item 1>
Prime Minister of Australia
Fraser accepted these terms and immediately advised a double dissolution . An election was called for 13 December , which the Liberal Party won in its own right
( although the Liberals governed in a coalition with the Country Party ) . Powers and role ( edit ) The first Prime Minister of Australia , Edmund Barton (
sitting second from left ) , with his Cabinet , 1901 . Most of the Prime Minister 's powers derive from being head of Government . In practice , the Federal
Executive Council will act to ratify all decisions made by the cabinet and decisions of the cabinet will always require the support of the Prime Minister . The
powers of the governor - general to grant Royal Assent to legislation , to dissolve and prorogue parliament , to call elections and to make appointments are
exercised on the advice of the Prime Minister .
</item 1>

<item 2>
Prime Minister of Australia
Vic , 1934 -- 1966 ( resigned ) -- 1st Menzies 2nd Menzies 1940 3rd Menzies 13 Arthur Fadden ( 1894 -- 1973 ) Country ( Coalition ) 28 August 1941 7 October
1941 Darling Downs , Qld 1936 -- 1949 McPherson , Qld 1949 -- 1958 ( retired ) -- Fadden 14 John Curtin ( 1885 -- 1945 ) Labor 7 October 1941 5 July 1945 †
Fremantle , WA , 1928 -- 1931 ( defeated ) ; 1934 -- 1945 ( died ) -- 1st Curtin 1943 2nd Curtin 15 Frank Forde ( 1890 -- 1983 ) Labor 6 July 1945 13 July 1945
Capricornia , Qld , 1922 -- 1946 ( defeated ) -- Forde 16 Ben Chifley ( 1885 -- 1951 ) Labor 13 July 1945 19 December 1949 Macquarie , NSW ,
</item 2>

<item 3>
Australia’s prime ministership began in 1901 and is based on convention, not explicitly defined in the Constitution. The Governor-General appoints the person
able to command House of Representatives confidence; ministers must be parliamentarians within three months. The Prime Minister leads Cabinet and government,
with The Lodge as usual residence. The material surveys leaders, powers, conventions, salaries, support and history, including brief caretaker governments,
leadership changes, the 1975 dismissal, and Harold Holt’s disappearance. Robert Men
</item 3>

<item 4>
Prime Minister of Australia
The leaders of the three parties , Alfred Deakin , George Reid and Chris Watson each served as Prime Minister before losing a vote of confidence . As a result
of the Labor Party 's split over conscription , Billy Hughes and his supporters were expelled from the Labor Party in November 1916 . He subsequently continued
on as prime minister at the head of the new National Labor Party , which had only 14 members out of a total of 75 in the House of Representatives . The
Commonwealth Liberal Party -- despite still forming the Official Opposition -- provided confidence and supply until February 1917 , when the two parties agree
to merge and form the Nationalist Party . During the 1975 constitutional crisis , on 11 November 1975 , the governor - general , Sir John Kerr , dismissed the
Labor Party 's Gough Whitlam as Prime Minister . Despite Labor holding a majority in the House of Representatives , Kerr appointed the Leader of the Opposition
, Liberal leader Malcolm Fraser as caretaker Prime Minister , conditional on the passage of the Whitlam government 's Supply bills through the Senate and the
calling of an election for both houses of parliament . Fraser accepted these terms and immediately advised a double dissolution .
</item 4>

<item 5>
Prime Minister of Australia
The governor - general , Lord Casey , commissioned the Leader of the Country Party , John McEwen , to form a government until the Liberal Party elected a new
leader . McEwen was Prime Minister for 23 days , until the election of ( then Senator ) John Gorton . Excluding the brief transition periods during changes of
government or leadership elections , there have only been a handful of cases where someone other than the leader of the majority party in the House of
Representatives was Prime Minister : Federation occurred on 1 January 1901 , but elections for the first parliament were not scheduled until late March . In the
interim , an unelected caretaker government was necessary . In what is now known as the Hopetoun Blunder , the governor - general , Lord Hopetoun , invited Sir
William Lyne , the premier of the most populous state , New South Wales , to form a government . Lyne was unable to do so and returned his commission in favour
of Edmund Barton , who became the first Prime Minister and led the inaugural government into and beyond the election . During the second parliament , three
parties ( Free Trade , Protectionist and Labor ) had roughly equal representation in the House of Representatives .
</item 5>

<item 6>
Australia’s prime ministers came from several parties, often leading coalitions. Earle Page and Arthur Fadden briefly became PM when their coalition’s larger
party lacked a leader; Fadden’s government fell after independents rejected its budget. After John Curtin died in 1945, Frank Forde served eight days before Ben
Chifley replaced him, making Forde the shortest-serving PM. Harold Holt disappeared while swimming in 1967 and was presumed dead. The roster traces Australian
leadership from Edmund Barton in
</item 6>

<item 7>
Prime Minister of Australia
and Administrators Premiers and Chief Ministers Parliaments and Assemblies State courts State electoral systems Local government ( show ) NSW Vic Qld WA SA Tas
NT Related topics ( show ) Regions Foreign relations Defence Force Public holidays Republicanism Other countries Atlas Australia 's first Prime Minister ,
Edmund Barton at the central table in the House of Representatives in 1901 . The Prime Minister of Australia is appointed by the Governor - General of Australia
under Section 64 of the Australian Constitution , which empowers the Governor - General , as the official representative of the monarch , to appoint government
ministers of state and requires them to be members of the House of Representatives or the Senate , or become members within three months of the appointment .
The Prime Minister and Treasurer are traditionally members of the House , but the Constitution does not have such a requirement . Before being sworn in as a
Minister of the Crown , a person must first be sworn in as a member of the Federal Executive Council if they are not already a member . Membership of the
Federal Executive Council entitles the member to the style of The Honourable ( usually abbreviated to The Hon ) for life , barring exceptional circumstances .
The senior members of the Executive Council constitute the Cabinet of Australia .
</item 7>

<item 8>
The Australian Prime Minister is the country’s head of government, appointed by the Governor-General by convention when able to command confidence in the House
of Representatives. The office began in 1901 with Edmund Barton; its term depends on maintaining that confidence. The article’s 2018 revision lists Malcolm
Turnbull, in office since 15 September 2015, and Deputy Prime Minister Michael McCormack. The official residence is The Lodge in Canberra, with Kirribilli House
also used. The annual salary was A
</item 8>

<item 9>
Prime Minister of Australia
He served under the United Australia Party and the Liberal Party respectively . The shortest - serving Prime Minister was Frank Forde , who was appointed to the
position on 6 July 1945 after the death of John Curtin , and served until 13 July 1945 when Ben Chifley was elected leader of the Australian Labor Party .
Parties Australian Labor Party Liberal Party of Australia Australian Country Party Nationalist Party of Australia United Australia Party Commonwealth Liberal
Party National Labor Party Free Trade Party Protectionist Party No . Name ( birth -- death ) Portrait Party Term of office Electorate served Elections won
Ministry Ref Sir Edmund Barton ( 1849 -- 1920 ) Protectionist 1 January 1901 24 September 1903 Hunter , NSW , 1901 -- 1903 ( resigned ) 1901 Barton Alfred
Deakin ( 1856 -- 1919 ) Protectionist 24 September 1903 27 April 1904 Ballaarat , Vic , 1901 -- 1913 ( retired ) 1903 1st Deakin Chris Watson ( 1867 -- 1941 )
Labour 27 April 1904 18 August 1904 Bland , NSW ,
</item 9>

<item 10>
Prime Minister of Australia
The formal title of the portfolio has always been simply `` Prime Minister '' , except for the period of the Fourth Deakin Ministry ( June 1909 to April 1910 )
, when it was known as `` Prime Minister ( without portfolio ) '' . If a government can not get its appropriation ( budget ) legislation passed by the House of
Representatives , or the House passes a vote of `` no confidence '' in the government , the Prime Minister is bound by convention to immediately advise the
Governor - General to dissolve the House of Representatives and hold a fresh election . Following a resignation in other circumstances , or the death of a Prime
Minister , the governor - general will generally appoint as Prime Minister the person elected as leader by the governing party or , in the case of a coalition ,
the senior party in the coalition . There have been four notable exceptions to this : When Joseph Lyons , Prime Minister and leader of the United Australia
Party ( UAP ) , died suddenly in April 1939 , the governor - general , Lord Gowrie , called on Sir Earle Page to become caretaker Prime Minister . Page was the
leader of the smaller party in the governing coalition , the Country Party .
</item 10>
=== END ITEM id=f439326c78f9d48b ===

=== ITEM id=3f1a789fb134e2b9 ===
Gold sentences and facts:
Sentence 0: The Rolling Stones The Rolling Stones performing at Summerfest in Milwaukee in 2015 . Left to right : Charlie Watts , Ronnie Wood , Mick Jagger and Keith Richards Background information Origin London , England Genres Rock blues Years active 1962 -- present Labels Decca London Rolling Stones Virgin ABKCO Interscope Polydor Associated acts The New Barbarians Billy Preston Faces The Pretty Things John Mayall & the Bluesbreakers Bill Wyman 's Rhythm Kings Website rollingstones.com Members Mick Jagger Keith Richards Charlie Watts Ronnie Wood Past members Brian Jones Ian Stewart Dick Taylor Bill Wyman Mick Taylor
  Fact 0.0: The Rolling Stones performed at Summerfest in Milwaukee in 2015.
  Fact 0.1: The photo shows, from left to right, Charlie Watts, Ronnie Wood, Mick Jagger and Keith Richards.
  Fact 0.2: The Rolling Stones have origin London, England.
  Fact 0.3: The Rolling Stones' genres are rock and blues.
  Fact 0.4: The Rolling Stones have been active from 1962 to the present.
  Fact 0.5: The Rolling Stones' labels include Decca, London, Rolling Stones, Virgin, ABKCO, Interscope, and Polydor.
  Fact 0.6: The Rolling Stones' associated acts include The New Barbarians, Billy Preston, Faces, The Pretty Things, John Mayall & the Bluesbreakers, and Bill Wyman's Rhythm Kings.
  Fact 0.7: The Rolling Stones' website is rollingstones.com.
  Fact 0.8: The current members of the Rolling Stones are Mick Jagger, Keith Richards, Charlie Watts, and Ronnie Wood.
  Fact 0.9: The past members of the Rolling Stones include Brian Jones, Ian Stewart, Dick Taylor, Bill Wyman, and Mick Taylor.
Sentence 1: Mick Jagger -- vocals , harmonica , rhythm guitar , percussion , keyboards ( 1962 -- present ) Keith Richards -- rhythm and lead guitar , bass guitar , vocals ( 1962 -- present ) Charlie Watts -- drums , percussion ( 1963 -- present ) Ronnie Wood -- lead and rhythm guitar , bass guitar , backing vocals ( 1975 -- present )
  Fact 1.0: Mick Jagger plays vocals, harmonica, rhythm guitar, percussion, and keyboards, and has been active from 1962 to the present.
  Fact 1.1: Keith Richards plays rhythm and lead guitar, bass guitar, and vocals, and has been active from 1962 to the present.
  Fact 1.2: Charlie Watts plays drums and percussion, and has been active from 1963 to the present.
  Fact 1.3: Ronnie Wood plays lead and rhythm guitar, bass guitar, and backing vocals, and has been active from 1975 to the present.
Sentence 2: The Rolling Stones are an English rock band formed in London , England in 1962 . The first stable line - up consisted of Brian Jones ( guitar , harmonica ) , Mick Jagger ( lead vocals ) , Keith Richards ( guitar , backing vocals ) , Bill Wyman ( bass ) , Charlie Watts ( drums ) , and Ian Stewart ( piano ) . Stewart was removed from the official line - up in 1963 but continued as a touring member until his death in 1985 . Jones left the band less than a month before his death in 1969 , having already been replaced by Mick Taylor , who remained until 1974 . After Taylor left the band , Ronnie Wood took his place in 1975 and has been on guitar in tandem with Richards ever since . Following Wyman 's departure in 1993 , Darryl Jones joined as their touring bassist . Touring keyboardists for the band have been Nicky Hopkins ( 1967 -- 1982 ) , Ian McLagan ( 1978 -- 1981 ) , Billy Preston ( through the mid-1970s ) and Chuck Leavell ( 1982 -- present ) . The band was first led by Brian Jones , but after developing into the band 's songwriters , Jagger and Richards assumed leadership when Jones was dealing with personal troubles and legal issues .
  Fact 2.0: The Rolling Stones are an English rock band.
  Fact 2.1: The Rolling Stones were formed in London, England in 1962.
  Fact 2.2: The first stable line-up of the Rolling Stones consisted of Brian Jones, Mick Jagger, Keith Richards, Bill Wyman, Charlie Watts, and Ian Stewart.
  Fact 2.3: Brian Jones played guitar and harmonica in the first stable line-up.
  Fact 2.4: Mick Jagger was the lead vocalist in the first stable line-up.
  Fact 2.5: Keith Richards played guitar and backing vocals in the first stable line-up.
  Fact 2.6: Bill Wyman played bass in the first stable line-up.
  Fact 2.7: Charlie Watts played drums in the first stable line-up.
  Fact 2.8: Ian Stewart played piano in the first stable line-up.
  Fact 2.9: Ian Stewart was removed from the official line-up in 1963.
  Fact 2.10: Ian Stewart continued as a touring member of the Rolling Stones until his death in 1985.
  Fact 2.11: Brian Jones left the Rolling Stones less than a month before his death in 1969.
  Fact 2.12: Mick Taylor replaced Brian Jones in the band.
  Fact 2.13: Mick Taylor remained in the band until 1974.
  Fact 2.14: Ronnie Wood took Mick Taylor's place in 1975 after Taylor left the band.
  Fact 2.15: Ronnie Wood has been on guitar in tandem with Keith Richards ever since 1975.
  Fact 2.16: Darryl Jones joined the Rolling Stones as their touring bassist after Bill Wyman's departure in 1993.
  Fact 2.17: Bill Wyman departed from the Rolling Stones in 1993.
  Fact 2.18: Nicky Hopkins was a touring keyboardist for the Rolling Stones from 1967 to 1982.
  Fact 2.19: Ian McLagan was a touring keyboardist for the Rolling Stones from 1978 to 1981.
  Fact 2.20: Billy Preston was a touring keyboardist for the Rolling Stones through the mid-1970s.
  Fact 2.21: Chuck Leavell has been a touring keyboardist for the Rolling Stones since 1982.
  Fact 2.22: The band was first led by Brian Jones.
  Fact 2.23: Jagger and Richards became the band's songwriters.
  Fact 2.24: Jagger and Richards assumed leadership of the band when Brian Jones was dealing with personal troubles and legal issues.

Retrieved items:
<item 1>
The Rolling Stones
Musical development ( edit ) See also : Instruments played by the Rolling Stones A copy of `` Micawber '' , Keith Richards ' signature Telecaster model , in the
Fender Guitar Factory Museum The Rolling Stones have assimilated various musical genres into their own collective sound . Throughout the band 's career , their
musical contributions have been marked by a continual reference and reliance on musical styles including blues , psychedelia , R&B , country , folk , reggae ,
dance , and world music , exemplified by Jones ' collaboration with the Master Musicians of Jajouka , as well as traditional English styles that use stringed
instrumentation like harps . Brian Jones experimented with the use of non-traditional instruments such as the sitar and slide guitar in their early days . The
group started out covering early rock ' n ' roll and blues songs , and have never stopped playing live or recording cover songs . Jagger and Richards shared an
admiration of Jimmy Reed , Muddy Waters , Howlin ' Wolf , and Little Walter , which influenced Brian Jones , of whom Richards says , `` He was more into T -
Bone Walker and jazz blues stuff .
</item 1>

<item 2>
The Rolling Stones
The Rolling Stones - wikipedia The Rolling Stones Jump to : navigation , search `` Rolling Stones '' redirects here . For other uses , see Rolling Stones (
disambiguation ) . For the magazine , see Rolling Stone . The Rolling Stones The Rolling Stones performing at Summerfest in Milwaukee in 2015 . Left to right :
Charlie Watts , Ronnie Wood , Mick Jagger and Keith Richards Background information Origin London , England Genres Rock blues Years active 1962 -- present
Labels Decca London Rolling Stones Virgin ABKCO Interscope Polydor Associated acts The New Barbarians Billy Preston Faces The Pretty Things John Mayall & the
Bluesbreakers Bill Wyman 's Rhythm Kings Website rollingstones.com Members Mick Jagger Keith Richards Charlie Watts Ronnie Wood Past members Brian Jones Ian
Stewart Dick Taylor Bill Wyman Mick Taylor The Rolling Stones are an English rock band formed in London , England in 1962 . The first stable line - up consisted
of Brian Jones ( guitar , harmonica ) , Mick Jagger ( lead vocals ) , Keith Richards ( guitar , backing vocals ) , Bill Wyman ( bass ) , Charlie Watts ( drums )
, and Ian Stewart ( piano ) .
</item 2>

<item 3>
The Rolling Stones
bass guitar , vocals ( 1962 -- present ) Charlie Watts -- drums , percussion ( 1963 -- present ) Ronnie Wood -- lead and rhythm guitar , bass guitar , backing
vocals ( 1975 -- present ) Former members Brian Jones -- rhythm guitar , harmonica , keyboards , piano , backing vocals ( 1962 -- 1969 ; died 1969 ) Ian Stewart
-- keyboards , piano ( 1962 -- 1963 ; touring 1969 , 1975 -- 1976 , 1978 , 1981 -- 1982 ; died 1985 ) Mick Taylor -- lead guitar , bass guitar , backing vocals
( 1969 -- 1974 ; guest 1981 , 2012 -- 2014 ) Bill Wyman -- bass guitar , keyboards , piano , backing vocals ( 1962 -- 1993 ;
</item 3>

<item 4>
The Rolling Stones
Gomelsky , who had no written agreement with the band , was not consulted . Initially , Oldham tried to apply the strategy used by Brian Epstein , the Beatles '
manager : making the band members wear suits . Later , he changed his mind and imagined a band which contrasted with the Beatles , featuring unmatched clothing
, long hair , an unclean appearance . He later said he wanted to make the Stones `` a raunchy , gamy , unpredictable bunch of undesirables '' and to ``
establish that the Stones were threatening , uncouth and animalistic '' . Stewart left the official line - up , but remained road manager and touring
keyboardist . On Stewart 's decision , Oldham later said `` Well , he just does n't look the part , and six is too many for ( fans ) to remember the faces in
the picture . '' Later , Oldham reduced the ages of the band members in publicity to make them appear as teenagers . Decca Records , which had declined to enter
into a deal with the Beatles , gave the Rolling Stones a recording contract with favourable terms .
</item 4>

<item 5>
The Rolling Stones
4 1968 -- 1972 : `` Back to basics '' 1.5 1972 -- 1977 : Critical fluctuations and Ronnie Wood 1.6 1978 -- 1982 : Commercial peak 1.7 1983 -- 1988 : Band
turmoil and solo efforts 1.8 1989 -- 1999 : Comeback , return to popularity , and record - breaking tours 1.9 2000 -- 2011 : A Bigger Bang and continued success
1.10 2012 -- present : 50th anniversary and covers album 2 Musical development 3 Legacy 4 Tours 5 Band members 6 Discography 7 See also 8 Notes 9 References 9.1
Footnotes 9.2 Sources 10 Further reading 11 External links History ( edit ) Early History ( edit ) Keith Richards and Mick Jagger became childhood friends and
classmates in 1950 in Dartford , Kent , before the Jagger family moved to Wilmington , five miles ( 8.05 km ) away , in 1954 .
</item 5>

<item 6>
The Rolling Stones
1981 -- 2014 ; died 2014 ) Jim Price -- trumpet ( 1970 -- 1973 ) Lisa Fischer -- backing vocals , co-lead vocals on `` Gimme Shelter '' ( 1989 -- 2015 ) Billy
Preston -- keyboards , backing vocals ( 1973 -- 1977 ; died 2006 ) Nicky Hopkins -- keyboards ( 1971 -- 1973 ; died 1994 ) Ernie Watts -- saxophone ( 1981 ) Ian
McLagan -- keyboards ( 1978 -- 1981 ; died 2014 ) Blondie Chaplin -- additional guitar , backing vocals ( 1997 -- 2007 ) Ollie E. Brown -- percussion ( 1975 --
1976 ) Timeline Discography ( edit ) Main article : The Rolling Stones discography Studio albums The Rolling Stones ( 1964 , UK ) / England 's Newest Hit Makers
( 1964 , US ) 12 X 5 ( 1964 , US ) The Rolling Stones No. 2 ( 1965 , UK ) / The Rolling Stones , Now !
</item 6>

<item 7>
The Rolling Stones
Stewart was removed from the official line - up in 1963 but continued as a touring member until his death in 1985 . Jones left the band less than a month before
his death in 1969 , having already been replaced by Mick Taylor , who remained until 1974 . After Taylor left the band , Ronnie Wood took his place in 1975 and
has been on guitar in tandem with Richards ever since . Following Wyman 's departure in 1993 , Darryl Jones joined as their touring bassist . Touring
keyboardists for the band have been Nicky Hopkins ( 1967 -- 1982 ) , Ian McLagan ( 1978 -- 1981 ) , Billy Preston ( through the mid-1970s ) and Chuck Leavell (
1982 -- present ) . The band was first led by Brian Jones , but after developing into the band 's songwriters , Jagger and Richards assumed leadership when
Jones was dealing with personal troubles and legal issues .
</item 7>

<item 8>
The Rolling Stones
By 1963 they were finding their musical stride as well as popularity , and in 1964 two unscientific opinion polls rated them as Britain 's most popular group ,
even outranking the Beatles . The name of the band was changed shortly after their first gig to `` The Rolling Stones '' . The group 's then acting manager
Giorgio Gomelsky secured a Sunday afternoon residency at the Crawdaddy Club in Richmond in February 1963 , which , Gomelsky claimed , triggered an ``
international renaissance for the blues '' . In May 1963 , The Rolling Stones signed Andrew Loog Oldham as their manager ; he was a former publicist who had
been directed to the band by his previous clients , the Beatles . Because Oldham was only nineteen and had not reached the age of majority -- he was also
younger than anyone in the band -- he could not get an agent 's licence nor sign any contracts without his mother also signing . By necessity he joined with
booking agent Eric Easton in order to secure record financing and assistance booking venues . Gomelsky , who had no written agreement with the band , was not
consulted .
</item 8>

<item 9>
The Rolling Stones
According to Richards , Jones named the band during a phone call to Jazz News . When asked by a journalist for the band 's name , Jones saw a Muddy Waters LP
lying on the floor ; one of the tracks was `` Rollin ' Stone '' . 1962 -- 1964 : building a following ( edit ) The back room of what was the Crawdaddy Club in
Richmond , London where the Rolling Stones had their first residency in 1963 Jones , Jagger , Richards , Stewart and Taylor played a gig billed as `` the Rollin
' Stones '' on 12 July 1962 , at the Marquee Club in London . Shortly afterwards the band went on their first tour of the UK , which they called a `` training
ground '' tour , because it was a new experience for all of them . Their material included the Chicago blues as well as Chuck Berry and Bo Diddley songs . At
that time , the line - up included neither bassist Bill Wyman , who joined in December 1962 , nor drummer Charlie Watts , who joined in January 1963 , thus
completing the band 's original rhythm section .
</item 9>

<item 10>
The Rolling Stones
keyboards , piano , backing vocals ( 1962 -- 1993 ; guest 2012 ) Early members Dick Taylor -- bass guitar ( 1962 ) Ricky Fenson -- bass guitar ( 1962 -- 1963 )
Colin Golding -- bass guitar ( 1962 -- 1963 ) Tony Chapman -- drums ( 1962 -- 1963 ) Carlo Little -- drums ( 1962 -- 1963 ) Current touring members Chuck
Leavell -- keyboards , backing vocals ( 1982 -- present ) Bernard Fowler -- backing vocals , percussion ( 1989 -- present ) Darryl Jones -- bass guitar ,
backing vocals ( 1993 -- present ) Matt Clifford -- keyboards , French horn , musical integrator ( 1989 -- 1990 , 2012 -- present ) Tim Ries -- saxophone ,
keyboards ( 1999 -- present ) Karl Denson -- saxophone ( 2014 -- present ) Sasha Allen -- backing vocals , co-lead vocals on `` Gimme Shelter '' ( 2016 --
present ) Former touring musicians Bobby Keys -- saxophone ( 1969 -- 1973 ; 1981 -- 2014 ;
</item 10>
=== END ITEM id=3f1a789fb134e2b9 ===
