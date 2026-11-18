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

This file has 6 items.

=== ITEM id=5eba6f223606cde7 ===
Gold sentences and facts:
Sentence 0: Transcription occurs in the cell nucleus , where the DNA is held . The DNA structure of the cell is made up of two helixes made up of sugar and phosphate held together by hydrogen bonds between the bases of opposite strands . The sugar and the phosphate in each strand are joined together by stronger phosphodiester covalent bonds . The DNA is `` unzipped '' ( disruption of hydrogen bonds between different single strands ) by the enzyme helicase , leaving the single nucleotide chain open to be copied . RNA polymerase reads the DNA strand from the 3 - prime ( 3 ' ) end to the 5 - prime ( 5 ' ) end , while it synthesizes a single strand of messenger RNA in the 5 ' - to - 3 ' direction . The general RNA structure is very similar to the DNA structure , but in RNA the nucleotide uracil takes the place that thymine occupies in DNA . The single strand of mRNA leaves the nucleus through nuclear pores , and migrates into the cytoplasm .
  Fact 0.0: Transcription occurs in the cell nucleus.
  Fact 0.1: The DNA is held in the cell nucleus.
  Fact 0.2: The DNA structure of the cell is made up of two helixes.
  Fact 0.3: Each helix of the DNA structure is made up of sugar and phosphate.
  Fact 0.4: The two helixes of the DNA structure are held together by hydrogen bonds between the bases of opposite strands.
  Fact 0.5: The sugar and the phosphate in each strand of DNA are joined together by phosphodiester covalent bonds.
  Fact 0.6: The phosphodiester covalent bonds are stronger than the hydrogen bonds.
  Fact 0.7: The DNA is "unzipped" by the enzyme helicase.
  Fact 0.8: The unzipping of the DNA is disruption of hydrogen bonds between different single strands.
  Fact 0.9: Unzipping the DNA leaves the single nucleotide chain open to be copied.
  Fact 0.10: RNA polymerase reads the DNA strand from the 3' end to the 5' end.
  Fact 0.11: While reading the DNA strand, RNA polymerase synthesizes a single strand of messenger RNA.
  Fact 0.12: RNA polymerase synthesizes the messenger RNA in the 5' to 3' direction.
  Fact 0.13: The general RNA structure is very similar to the DNA structure.
  Fact 0.14: In RNA, the nucleotide uracil takes the place that thymine occupies in DNA.
  Fact 0.15: The single strand of messenger RNA leaves the nucleus through nuclear pores.
  Fact 0.16: The single strand of messenger RNA migrates into the cytoplasm.

Retrieved items:
<item 1>
Protein biosynthesis
The first product of transcription differs in prokaryotic cells from that of eukaryotic cells , as in prokaryotic cells the product is mRNA , which needs no
post-transcriptional modification , whereas , in eukaryotic cells , the first product is called primary transcript , that needs post-transcriptional
modification ( capping with 7 - methyl - guanosine , tailing with a poly A tail ) to give hnRNA ( heterogeneous nuclear RNA ) . hnRNA then undergoes splicing of
introns ( noncoding parts of the gene ) via spliceosomes to produce the final mRNA . Translation ( edit ) Main article : Translation ( biology ) Diagram showing
the process of translation Diagram showing the translation of mRNA and the synthesis of proteins by a ribosome Phenomena of amino acid assembly from RNA . The
synthesis of proteins from RNA is known as translation . In eukaryotes , translation occurs in the cytoplasm , where the ribosomes are located . Ribosomes are
made of a small and large subunit that surround the mRNA .
</item 1>

<item 2>
Protein biosynthesis
Ribosomes are made of a small and large subunit that surround the mRNA . In translation , messenger RNA ( mRNA ) is decoded to produce a specific polypeptide
according to the rules specified by the trinucleotide genetic code . This uses an mRNA sequence as a template to guide the synthesis of a chain of amino acids
that form a protein . Translation proceeds in four phases : activation , initiation , elongation , and termination ( all describing the growth of the amino acid
chain , or polypeptide that is the product of translation ) . In activation , the correct amino acid ( AA ) is joined to the correct transfer RNA ( tRNA ) .
While this is not , in the technical sense , a step in translation , it is required for translation to proceed . The AA is joined by its carboxyl group to the 3
' OH of the tRNA by an ester bond . When the tRNA has an amino acid linked to it , it is termed `` charged '' . Initiation involves the small subunit of the
ribosome binding to 5 ' end of mRNA with the help of initiation factors ( IF ) , other proteins that assist the process .
</item 2>

<item 3>
Protein biosynthesis
Serine -- Tyrosine -- Glycine p - Hydroxybenzylidene - imidazolinone formation Histidine -- Tyrosine -- Glycine 4 - ( p - hydroxybenzylidene ) - 5 -
imidazolinone formation Crosslinks between four AAs Allysine - Allysine - Allysine - Lysine Desmosine Gene expression Introduction to genetics Genetic code
Central dogma DNA → RNA → Protein Special transfers RNA → RNA RNA → DNA Protein → Protein Transcription Types Bacterial Eukaryotic Key elements Transcription
factor RNA polymerase Promoter Post-transcription Precursor mRNA ( pre-mRNA / hnRNA ) 5 ' capping Splicing Polyadenylation Histone acetylation and deacetylation
Translation Types Prokaryotic Eukaryotic Key elements Ribosome Transfer RNA ( tRNA ) Ribosome - nascent chain complex ( RNC ) Post-translational modification (
functional groups peptides structural changes ) Regulation Epigenetic imprinting Transcriptional Gene regulatory network cis - regulatory element lac operon
Post-transcriptional sequestration ( P - bodies ) alternative splicing microRNA Translational Post-translational reversible irreversible
</item 3>

<item 4>
Protein biosynthesis
Protein biosynthesis - wikipedia Protein biosynthesis Jump to : navigation , search RNA is transcribed in the nucleus ; once completely processed , it is
transported to the cytoplasm and translated by the ribosome ( shown in very pale grey behind the tRNA ) . Protein synthesis is the process whereby biological
cells generate new proteins ; it is balanced by the loss of cellular proteins via degradation or export . Translation , the assembly of amino acids by ribosomes
, is an essential part of the biosynthetic pathway , along with generation of messenger RNA ( mRNA ) , aminoacylation of transfer RNA ( tRNA ) ,
co-translational transport , and post-translational modification . Protein biosynthesis is strictly regulated at multiple steps . They are principally during
transcription ( phenomena of RNA synthesis from DNA template ) and translation ( phenomena of amino acid assembly from RNA ) . The cistron DNA is transcribed
into the first of a series of RNA intermediates . The last version is used as a template in synthesis of a polypeptide chain . Protein will often be synthesized
directly from genes by translating mRNA .
</item 4>

<item 5>
Protein biosynthesis
The termini and side - chains of the polypeptide may be subjected to post-translational modification . These modifications may be required for correct cellular
localisation or the natural function of the protein . During and after synthesis , polypeptide chains often fold to assume , so called , native secondary and
tertiary structures . This is known as protein folding and is typically required for the natural function of the protein . See also ( edit ) Central dogma of
molecular biology Cistron Gene expression Genetic code Operon Peptide synthesis Protein production References ( edit ) Jump up ^ Kafri M , Metzl - Raz E , Jona
G , Barkai N. 2016 . The Cost of Protein Production . Cell Rep 14 : 22 -- 31 . https://dx.doi.org/10.1016/j.celrep.2015.12.015 ^ Jump up to : Alberts , Bruce (
2002 ) . Molecular biology of the cell . New York : Garland Science . p. 760 . ISBN 0 - 8153 - 3218 - 1 . Jump up ^ Alberts , Bruce .
</item 5>

<item 6>
Protein biosynthesis
This whole complex of processes is carried out by the ribosome , formed of two main chains of RNA , called ribosomal RNA ( rRNA ) , and more than 50 different
proteins . The ribosome latches onto the end of an mRNA molecule and moves along it , capturing loaded tRNA molecules and joining together their amino acids to
form a new protein chain . Protein biosynthesis , although very similar , is different for prokaryotes and eukaryotes . Contents ( hide ) 1 Transcription 2
Translation 3 Events during or following protein translation 4 See also 5 References 6 External links Transcription ( edit ) Main article : Transcription (
genetics ) Diagram showing the process of transcription Phenomena of RNA synthesis from DNA template In transcription an mRNA chain is generated , with one
strand of the DNA double helix in the genome as a template . This strand is called the template strand . Transcription can be divided into 3 stages : initiation
, elongation , and termination , each regulated by a large number of proteins such as transcription factors and coactivators that ensure that the correct gene
is transcribed . Transcription occurs in the cell nucleus , where the DNA is held .
</item 6>

<item 7>
Protein biosynthesis
Elongation occurs when the next aminoacyl - tRNA ( charged tRNA ) in line binds to the ribosome along with GTP and an elongation factor . Termination of the
polypeptide happens when the A site of the ribosome faces a stop codon ( UAA , UAG , or UGA ) . When this happens , no tRNA can recognize it , but releasing
factor can recognize nonsense codons and causes the release of the polypeptide chain . The capacity of disabling or inhibiting translation in protein
biosynthesis is used by some antibiotics such as anisomycin , cycloheximide , chloramphenicol , tetracycline , streptomycin , erythromycin , puromycin , etc .
Events during or following protein translation ( edit ) Main articles : Proteolysis , Posttranslational modification , and Protein folding Events that occur
during or following biosynthesis include proteolysis , post-translational modification and protein folding . Proteolysis may remove N - terminal , C - terminal
or internal amino - acid residues or peptides from the polypeptide .
</item 7>

<item 8>
Protein biosynthesis
Transcription occurs in the cell nucleus , where the DNA is held . The DNA structure of the cell is made up of two helixes made up of sugar and phosphate held
together by hydrogen bonds between the bases of opposite strands . The sugar and the phosphate in each strand are joined together by stronger phosphodiester
covalent bonds . The DNA is `` unzipped '' ( disruption of hydrogen bonds between different single strands ) by the enzyme helicase , leaving the single
nucleotide chain open to be copied . RNA polymerase reads the DNA strand from the 3 - prime ( 3 ' ) end to the 5 - prime ( 5 ' ) end , while it synthesizes a
single strand of messenger RNA in the 5 ' - to - 3 ' direction . The general RNA structure is very similar to the DNA structure , but in RNA the nucleotide
uracil takes the place that thymine occupies in DNA . The single strand of mRNA leaves the nucleus through nuclear pores , and migrates into the cytoplasm .
</item 8>

<item 9>
Protein biosynthesis
Protein will often be synthesized directly from genes by translating mRNA . However , when a protein must be available on short notice or in large quantities ,
a protein precursor is produced . A proprotein is an inactive protein containing one or more inhibitory peptides that can be activated when the inhibitory
sequence is removed by proteolysis during posttranslational modification . A preprotein is a form that contains a signal sequence ( an N - terminal signal
peptide ) that specifies its insertion into or through membranes , i.e. , targets them for secretion . The signal peptide is cleaved off in the endoplasmic
reticulum . Preproproteins have both sequences ( inhibitory and signal ) still present . In protein synthesis , a succession of tRNA molecules charged with
appropriate amino acids are brought together with an mRNA molecule and matched up by base - pairing through the anti-codons of the tRNA with successive codons
of the mRNA . The amino acids are then linked together to extend the growing protein chain , and the tRNAs , no longer carrying amino acids , are released .
</item 9>

<item 10>
Protein biosynthesis
Jump up ^ Alberts , Bruce . Molecular Biology of the Cell , 5e . New York : Garland Science , 2008 . External links ( edit ) Interactive Java simulation of
transcription initiation . From Center for Models of Life at the Niels Bohr Institute . A website focused primarily on Protein Synthesis . Protein primary
structure and posttranslational modifications General Peptide bond Protein biosynthesis Proteolysis Racemization N-O acyl shift N terminus Acetylation
Carbamylation Formylation Glycation Methylation Myristoylation ( Gly ) C terminus Amidation Glycosyl phosphatidylinositol ( GPI ) O - methylation Detyrosination
Single specific AAs Serine / Threonine Phosphorylation Dephosphorylation Glycosylation Methylidene - imidazolone ( MIO ) formation Tyrosine Phosphorylation
Dephosphorylation Sulfation Porphyrin ring linkage Adenylylation Flavin linkage Topaquinone ( TPQ ) formation Detyrosination Cysteine Palmitoylation Prenylation
Aspartate Succinimide formation Glutamate Carboxylation Methylation
</item 10>
=== END ITEM id=5eba6f223606cde7 ===

=== ITEM id=93185214091729a4 ===
Gold sentences and facts:
Sentence 0: The original start date was January 2002 , but was pushed to February 7 in Los Angeles , California . Locations included Burbank , Downey , New York City , LA / Ontario International Airport ( which doubled for Miami International Airport ) , Quebec City and Montreal . The film was shot in 147 locations in only 52 days . DiCaprio reflected , `` Scenes that we thought would take three days took an afternoon '' . Filming ran from April 25 -- 30 in Park Avenue , just outside the Waldorf - Astoria Hotel . Production moved to Orange , New Jersey and returned to Brooklyn for bank and courthouse scenes . Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport . Quebec City was chosen for its European character and French feel . Place Royale , within Old Quebec , stands for Montrichard -- the church in the background of the arrest scene is Notre - Dame - des - Victoires . Filming ended on May 12 in Montreal .
  Fact 0.0: The original start date was January 2002.
  Fact 0.1: The start date was pushed to February 7 in Los Angeles, California.
  Fact 0.2: Locations included Burbank.
  Fact 0.3: Locations included Downey.
  Fact 0.4: Locations included New York City.
  Fact 0.5: Locations included LA / Ontario International Airport.
  Fact 0.6: LA / Ontario International Airport doubled for Miami International Airport.
  Fact 0.7: Locations included Quebec City.
  Fact 0.8: Locations included Montreal.
  Fact 0.9: The film was shot in 147 locations.
  Fact 0.10: The film was shot in only 52 days.
  Fact 0.11: DiCaprio reflected that scenes thought to take three days took an afternoon.
  Fact 0.12: Filming ran from April 25 to 30 in Park Avenue.
  Fact 0.13: Park Avenue filming was just outside the Waldorf-Astoria Hotel.
  Fact 0.14: Production moved to Orange, New Jersey.
  Fact 0.15: Production returned to Brooklyn for bank and courthouse scenes.
  Fact 0.16: Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport.
  Fact 0.17: Quebec City was chosen for its European character.
  Fact 0.18: Quebec City was chosen for its French feel.
  Fact 0.19: Place Royale, within Old Quebec, stands for Montrichard.
  Fact 0.20: The church in the background of the arrest scene is Notre-Dame-des-Victoires.
  Fact 0.21: Filming ended on May 12 in Montreal.
Sentence 1: Catch Me If You Can is a 2002 American biographical crime film directed and co-produced by Steven Spielberg from a screenplay by Jeff Nathanson . The film is based on the life of Frank Abagnale , who , before his 19th birthday , successfully performed cons worth millions of dollars by posing as a Pan American World Airways pilot , a Georgia doctor and a Louisiana parish prosecutor . His primary crime was check fraud ; he became so experienced that the FBI eventually turned to him for help in catching other checking forgers . The film stars Leonardo DiCaprio and Tom Hanks , with Christopher Walken , Martin Sheen , and Nathalie Baye in supporting roles .
  Fact 1.0: Catch Me If You Can is a 2002 American biographical crime film.
  Fact 1.1: Catch Me If You Can was directed by Steven Spielberg.
  Fact 1.2: Catch Me If You Can was co-produced by Steven Spielberg.
  Fact 1.3: Catch Me If You Can has a screenplay by Jeff Nathanson.
  Fact 1.4: The film is based on the life of Frank Abagnale.
  Fact 1.5: Frank Abagnale successfully performed cons worth millions of dollars before his 19th birthday.
  Fact 1.6: Frank Abagnale performed cons by posing as a Pan American World Airways pilot.
  Fact 1.7: Frank Abagnale performed cons by posing as a Georgia doctor.
  Fact 1.8: Frank Abagnale performed cons by posing as a Louisiana parish prosecutor.
  Fact 1.9: Abagnale's primary crime was check fraud.
  Fact 1.10: Abagnale became so experienced that the FBI eventually turned to him for help.
  Fact 1.11: The FBI turned to Abagnale for help in catching other checking forgers.
  Fact 1.12: The film stars Leonardo DiCaprio.
  Fact 1.13: The film stars Tom Hanks.
  Fact 1.14: Christopher Walken, Martin Sheen, and Nathalie Baye have supporting roles in the film.
Sentence 2: Development for the film started in 1980 , but did not progress until 1997 , when the film rights to Abagnale 's book were sold to Spielberg 's DreamWorks . David Fincher , Gore Verbinski , Lasse Hallström , Miloš Forman , and Cameron Crowe had all been possible candidates for director before Spielberg decided to direct . Filming took place from February to May 2002 . The film was a financial and critical success , and was received positively by both critics and Abagnale .
  Fact 2.0: Development for the film started in 1980.
  Fact 2.1: Development did not progress until 1997.
  Fact 2.2: In 1997, the film rights to Abagnale's book were sold to Spielberg's DreamWorks.
  Fact 2.3: David Fincher was a possible candidate for director before Spielberg decided to direct.
  Fact 2.4: Gore Verbinski was a possible candidate for director before Spielberg decided to direct.
  Fact 2.5: Lasse Hallström was a possible candidate for director before Spielberg decided to direct.
  Fact 2.6: Miloš Forman was a possible candidate for director before Spielberg decided to direct.
  Fact 2.7: Cameron Crowe was a possible candidate for director before Spielberg decided to direct.
  Fact 2.8: Filming took place from February to May 2002.
  Fact 2.9: The film was a financial success.
  Fact 2.10: The film was a critical success.
  Fact 2.11: The film was received positively by both critics and Abagnale.

Retrieved items:
<item 1>
Catch Me If You Can
Jump up ^ Mick LaSalle ( December 25 , 2002 ) . `` Holiday Movies '' . San Francisco Chronicle . Retrieved July 3 , 2008 . Jump up ^ Stephen Hunter ( December
25 , 2002 ) . `` A Merry Chase '' . The Washington Post . Retrieved July 3 , 2008 . Jump up ^ Berardinelli , James . `` Catch Me If You Can '' . ReelViews.net .
Retrieved July 3 , 2008 . Jump up ^ Travers , Peter ( January 2 , 2003 ) . `` Catch Me If You Can '' . Rolling Stone . Archived from the original on April 19 ,
2008 . Retrieved July 3 , 2008 . Jump up ^ `` 74th Academy Awards '' . Internet Movie Database . Retrieved July 3 , 2008 . Jump up ^ `` 56th BAFTA Awards '' .
Internet Movie Database . Retrieved July 3 , 2008 . Jump up ^ `` Golden Globes : 2003 '' . Internet Movie Database .
</item 1>

<item 2>
Catch Me If You Can
Jump up ^ Josef Adalian ( December 10 , 2002 ) . `` Inside Move : Net game for movie link '' . Variety . Retrieved June 30 , 2008 . Jump up ^ `` H'w 'd plays '
Catch ' '' . Variety . December 18 , 2002 . Retrieved June 30 , 2008 . Jump up ^ `` Catch Me If You Can ( 2002 ) '' . Box Office Mojo . Internet Movie Database
. Retrieved June 29 , 2008 . Jump up ^ `` 2002 Yearly Box Office Results '' . Box Office Mojo . Internet Movie Database . Retrieved June 29 , 2008 . Jump up ^
`` Catch Me If You Can '' . Rotten Tomatoes . Flixter . Retrieved June 29 , 2008 . Jump up ^ Ebert , Roger . `` Catch Me If You Can '' . Chicago Sun - Times .
Retrieved July 3 , 2008 . Jump up ^ Mick LaSalle ( December 25 , 2002 ) .
</item 2>

<item 3>
Catch Me If You Can
Jump up ^ Brian Linder ( May 2 , 2002 ) . `` Spielly Update : ' Report ' , ' Catch Me ' '' . IGN . Retrieved July 1 , 2008 . Jump up ^ Army Archerd ( May 14 ,
2002 ) . `` ' Sierra Madre ' inspires ' Spider - Man ' helmer '' . Variety . Retrieved July 1 , 2008 . ^ Jump up to : Stax ( November 15 , 2001 ) . `` The Man
Behind ' Catch Me If You Can ' '' . IGN . Retrieved July 1 , 2008 . ^ Jump up to : Steve Head ( December 17 , 2002 ) . `` An Interview with Steven Spielberg ''
. IGN . Retrieved June 30 , 2008 . Jump up ^ Claude Brodesser ( August 28 , 2001 ) . `` D'Works tracking top cop for Catch '' . Variety . Retrieved June 30 ,
2008 . Jump up ^ Josef Adalian ( December 10 , 2002 ) .
</item 3>

<item 4>
Catch Me If You Can
Catch Me If You Can - wikipedia Catch Me If You Can Jump to : navigation , search This article is about the 2002 film . For other uses , see Catch Me If You Can
( disambiguation ) . Catch Me If You Can Theatrical release poster Directed by Steven Spielberg Produced by Steven Spielberg Walter F. Parkes Screenplay by Jeff
Nathanson Based on Catch Me If You Can by Frank Abagnale and Stan Redding Starring Leonardo DiCaprio Tom Hanks Music by John Williams Cinematography Janusz
Kamiński Edited by Michael Kahn Production company Amblin Entertainment Parkes / MacDonald Productions Distributed by DreamWorks Pictures Release date December
25 , 2002 ( 2002 - 12 - 25 ) Running time 141 minutes Country United States Language English Budget $52 million Box office $352.1 million Catch Me If You Can is
a 2002 American biographical crime film directed and co-produced by Steven Spielberg from a screenplay by Jeff Nathanson .
</item 4>

<item 5>
Catch Me If You Can
Williams also earned a Grammy Award nomination . Elements of the film were later parodied in The Simpsons episode `` Catch ' Em If You Can '' . Home media (
edit ) Catch Me If You Can was released on DVD on May 6 , 2003 and on Blu - ray on December 4 , 2012 . Musical adaptation ( edit ) Main article : Catch Me If
You Can ( musical ) A musical adaptation of the same name premiered at the 5th Avenue Theatre in Seattle , Washington in July 2009 , starring Aaron Tveit and
Norbert Leo Butz . It began previews on Broadway at the Neil Simon Theatre on March 11 , 2011 and officially opened April 10 , 2011 . The musical was nominated
for four Tony Awards , including Best Musical . See also ( edit ) The Great Impostor , a 1961 film based on the story of an impostor named Ferdinand Waldo
Demara . The Pretender , a TV series References ( edit ) Jump up ^ `` CATCH ME IF YOU CAN ( 12A ) '' . British Board of Film Classification . December 13 , 2002
.
</item 5>

<item 6>
Catch Me If You Can
British Board of Film Classification . December 13 , 2002 . Retrieved February 13 , 2016 . Jump up ^ Van Luling , Todd ( October 17 , 2014 ) . `` 11 Easter Eggs
You Never Noticed In Your Favorite Movies '' . The Huffington Post . TheHuffingtonPost.com , Inc . Retrieved January 29 , 2015 . ^ Jump up to : Frank Abagnale (
September 3 , 2002 ) . `` Comments '' . Abagnale & Associates . Archived from the original on February 16 , 2009 . Retrieved June 30 , 2008 . ^ Jump up to :
Claude Brodesser ; Dana Harris ( August 21 , 2001 ) . `` D'Works to play ' Catch ' '' . Variety . Retrieved June 30 , 2008 . Jump up ^ Charles Lyons ; Dade
Hayes ( January 8 , 2002 ) . `` D'Works sets play date for pricey ' Catch ' '' . Variety . Retrieved June 29 , 2008 .
</item 6>

<item 7>
Catch Me If You Can
wikipedia.org/w/index.php?title=Catch_Me_If_You_Can&oldid=821436887 '' Categories : 2002 films English - language films 2000s biographical films 2000s crime
drama films Amblin Entertainment films American biographical films American chase films American crime drama films American films Biographical films about
fraudsters Crime films based on actual events Drama films based on actual events DreamWorks Pictures films Films scored by John Williams Films about con artists
Films about identity theft Films based on biographies Films directed by Steven Spielberg Films set in Atlanta Films set in California Films set in Florida Films
set in France Films set in Louisiana Films set in Westchester County , New York Films set in New York City Films set in the 1960s Films shot in California Films
shot in Montreal Films produced by Steven Spielberg True crime films Fraud in fiction Hidden categories : Webarchive template wayback links Use mdy dates from
October 2015 Good articles Wikipedia articles with GND identifiers Talk Contents About Wikipedia Wikiquote বাংলা Български Català Čeština Dansk Deutsch Español
Euskara فارسی Français Galego 한국어
</item 7>

<item 8>
Catch Me If You Can
Frank Abagnale , Jr. and Stan Redding . Catch Me If You Can : The Amazing True Story of the Youngest and Most Daring Con Man in the History of Fun and Profit .
( ISBN 0 - 06 - 052971 - 7 ) . External links ( edit ) Wikiquote has quotations related to : Catch Me If You Can Catch Me If You Can on IMDb Catch Me If You Can
at AllMovie Catch Me If You Can at the TCM Movie Database Catch Me If You Can at the American Film Institute Catalog Catch Me If You Can at Box Office Mojo
Catch Me If You Can at Rotten Tomatoes Catch Me If You Can at Metacritic An Interview with Leonardo DiCaprio - Movies Feature at IGN Official website for Catch
Me If You Can the musical ( hide ) Steven Spielberg Filmography Awards and nominations Films directed Firelight ( 1964 ) Slipstream ( 1967 ) Amblin ' ( 1968 )
Night Gallery ( `` Eyes '' segment , 1969 ) L.A.
</item 8>

<item 9>
Catch Me If You Can
The film is based on the life of Frank Abagnale , who , before his 19th birthday , successfully performed cons worth millions of dollars by posing as a Pan
American World Airways pilot , a Georgia doctor and a Louisiana parish prosecutor . His primary crime was check fraud ; he became so experienced that the FBI
eventually turned to him for help in catching other checking forgers . The film stars Leonardo DiCaprio and Tom Hanks , with Christopher Walken , Martin Sheen ,
and Nathalie Baye in supporting roles . Development for the film started in 1980 , but did not progress until 1997 , when the film rights to Abagnale 's book
were sold to Spielberg 's DreamWorks . David Fincher , Gore Verbinski , Lasse Hallström , Miloš Forman , and Cameron Crowe had all been possible candidates for
director before Spielberg decided to direct . Filming took place from February to May 2002 . The film was a financial and critical success , and was received
positively by both critics and Abagnale .
</item 9>

<item 10>
Catch Me If You Can
The marketing department was careful to market the film as `` inspired by a true story '' in order to avoid controversy similar to that surrounding A Beautiful
Mind ( 2001 ) and The Hurricane ( 1999 ) , both of which deviated from history . The premiere took place at Westwood , Los Angeles , California on December 18 ,
2002 . Box Office ( edit ) Catch Me If You Can was released on December 25 , 2002 , earning slightly above $30 million in 3,225 theaters during its opening
weekend . The film went on to gross $164.6 million in North America and $187.5 million in foreign countries , coming at a worldwide total of $352.1 million .
The film was a financial success , recouping the $52 million budget six times over . Catch Me If You Can was the eleventh highest - grossing film of 2002 .
Minority Report ( also directed by Spielberg ) was tenth highest . Critical response ( edit ) On Rotten Tomatoes , the film has a certified fresh rating of 96 %
, based on 197 reviews , with an average rating of 7.9 / 10 .
</item 10>
=== END ITEM id=93185214091729a4 ===

=== ITEM id=11494f8a6120a450 ===
Gold sentences and facts:
Sentence 0: Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name ( Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741 7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743 -- First Lord of the Treasury Whig Carteret 7002502000000000000 ♠ 1 year and 137 days ​ The Right Honourable Henry Pelham FRS MP for Sussex ( 1694 -- 1754 ) 27 August 1743 6 March 1754 1747 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Broad Bottom I Broad Bottom II 7003384500000000000 ♠ 10 years and 192 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 16 March 1754 16 November 1756 1754 First Lord of the Treasury Leader of the House of Lords Whig Newcastle I 7002977000000000000 ♠ 2 years and 246 days His Grace William Cavendish 4th Duke of Devonshire KG PC ( 1720 -- 1764 ) 16 November 1756 25 June 1757 -- First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Devonshire ​ 1757 Caretaker 7002222000000000000 ♠ 222 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 2 July 1757 26 May 1762 1761 First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Newcastle George III ( 1760 -- 1820 ) 7003179000000000000 ♠ 4 years and 329 days The Right Honourable John Stuart 3rd Earl of Bute KG PC ( 1713 -- 1792 ) 26 May 1762 8 April 1763 -- First Lord of the Treasury Leader of the House of Lords Tory Bute ( Tory -- Whig ) 7002318000000000000 ♠ 318 days The Right Honourable George Grenville MP for Buckingham ( 1712 -- 1770 ) 16 April 1763 13 July 1765 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig ( Grenvillite ) Grenville ( Whig -- Tory ) 7002820000000000000 ♠ 2 years and 89 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 13 July 1765 30 July 1766 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham I 7002383000000000000 ♠ 1 year and 18 days The Right Honourable William Pitt 1st Earl of Chatham PC FRS MP for Bath ( 1708 -- 1778 ) 30 July 1766 14 October 1768 1768 Lord Keeper of the Privy Seal Whig ( Chathamite ) Chatham ( Whig -- Tory ) 7002808000000000000 ♠ 2 years and 77 days His Grace Augustus FitzRoy 3rd Duke of Grafton KG PC ( 1735 -- 1811 ) 14 October 1768 28 January 1770 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Grafton 7002472000000000000 ♠ 1 year and 107 days The Right Honourable Frederick North Lord North KG MP for Banbury ( 1732 -- 1792 ) 28 January 1770 22 March 1782 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Northite ) North 1774 1780 7003443700000000000 ♠ 12 years and 54 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 27 March 1782 1 July 1782 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham II 7001970000000000000 ♠ 97 days The Right Honourable William Petty 2nd Earl of Shelburne KG PC ( 1737 -- 1805 ) 4 July 1782 2 April 1783 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Shelburne ( Whig -- Tory ) 7002273000000000000 ♠ 273 days His Grace William Cavendish - Bentinck 3rd Duke of Portland PC FRS ( 1738 -- 1809 ) 2 April 1783 19 December 1783 -- First Lord of the Treasury Leader of the House of Lords Whig Fox -- North 7002262000000000000 ♠ 262 days The Right Honourable William Pitt the Younger MP for Appleby -- Cambridge University ( 1759 -- 1806 ) 19 December 1783 14 March 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt I 1784 1790 1796 7003629500000000000 ♠ 17 years and 86 days The Right Honourable Henry Addington MP for Devizes ( 1757 -- 1844 ) 17 March 1801 10 May 1804 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Addington 1802 7003115100000000000 ♠ 3 years and 55 days The Right Honourable William Pitt the Younger MP for Cambridge University ( 1759 -- 1806 ) 10 May 1804 23 January 1806 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt II 7002624000000000000 ♠ 1 year and 259 days The Right Honourable William Grenville 1st Baron Grenville PC PC ( Ire ) ( 1759 -- 1834 ) 11 February 1806 31 March 1807 First Lord of the Treasury Leader of the House of Lords Whig All the Talents ( Whig -- Tory ) 1806 7002414000000000000 ♠ 1 year and 49 days His Grace William Cavendish - Bentinck 3rd Duke of Portland KG PC FRS ( 1738 -- 1809 ) 31 March 1807 4 October 1809 First Lord of the Treasury Tory ( Pittite ) Portland II 1807 7002919000000000000 ♠ 2 years and 188 days The Right Honourable Spencer Perceval KC MP for Northampton ( 1762 -- 1812 ) 4 October 1809 11 May 1812 -- First Lord of the Treasury Chancellor of the Exchequer Chancellor of the Duchy of Lancaster Leader of the House of Commons Tory ( Pittite ) Perceval 7002951000000000000 ♠ 2 years and 221 days The Right Honourable Robert Jenkinson 2nd Earl of Liverpool KG PC FRS ( 1770 -- 1828 ) 8 June 1812 9 April 1827 1812 First Lord of the Treasury Leader of the House of Lords Tory ( Pittite ) Liverpool 1818 1820 George IV ( 1820 -- 1830 ) 1826 7003541900000000000 ♠ 14 years and 306 days The Right Honourable George Canning FRS MP for Seaford ( 1770 -- 1827 ) 10 April 1827 8 August 1827 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Canningite ) Canning ( Canningite -- Whig ) 7002121000000000000 ♠ 121 days The Right Honourable Frederick John Robinson 1st Viscount Goderich PC ( 1782 -- 1859 ) 31 August 1827 21 January 1828 -- First Lord of the Treasury Leader of the House of Lords Tory ( Canningite ) Goderich 7002144000000000000 ♠ 144 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 22 January 1828 16 November 1830 First Lord of the Treasury Leader of the House of Lords Tory Wellington -- Peel 1830 William IV ( 1830 -- 1837 ) 7003103000000000000 ♠ 2 years and 299 days The Right Honourable Charles Grey 2nd Earl Grey KG PC ( 1764 -- 1845 ) 22 November 1830 9 July 1834 First Lord of the Treasury Leader of the House of Lords Whig Grey 1831 1832 7003132600000000000 ♠ 3 years and 230 days The Right Honourable William Lamb 2nd Viscount Melbourne PC ( 1779 -- 1848 ) 16 July 1834 14 November 1834 -- First Lord of the Treasury Leader of the House of Lords Whig Melbourne I 7002122000000000000 ♠ 122 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 14 November 1834 10 December 1834 -- First Lord of the Treasury Sec. of State for the Home Department Sec. of State for Foreign Affairs Sec. of State for War and the Colonies Leader of the House of Lords Tory Wellington Caretaker 7001270000000000000 ♠ 27 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 10 December 1834 8 April 1835 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Conservative Peel I 7002120000000000000 ♠ 120 days The Right Honourable William Lamb 2nd Viscount Melbourne PC FRS ( 1779 -- 1848 ) 18 April 1835 30 August 1841 1835 First Lord of the Treasury Leader of the House of Lords Whig Melbourne II 1837 Victoria ( 1837 -- 1901 ) 7003232700000000000 ♠ 6 years and 135 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 30 August 1841 29 June 1846 1841 First Lord of the Treasury Leader of the House of Commons Conservative Peel II 7003176500000000000 ♠ 4 years and 304 days The Right Honourable Lord John Russell FRS MP for the City of London ( 1792 -- 1878 ) 30 June 1846 21 February 1852 1847 First Lord of the Treasury Leader of the House of Commons Whig Russell I 7003206300000000000 ♠ 5 years and 237 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 23 February 1852 17 December 1852 First Lord of the Treasury Leader of the House of Lords Conservative Who ? Who ? 1852 7002299000000000000 ♠ 299 days The Right Honourable George Hamilton - Gordon 4th Earl of Aberdeen KT FRSE FRS PC FSA Scot ( 1784 -- 1860 ) 19 December 1852 30 January 1855 -- First Lord of the Treasury Leader of the House of Lords Peelite Aberdeen ( Peelite -- Whig ) 7002773000000000000 ♠ 2 years and 43 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 6 February 1855 19 February 1858 First Lord of the Treasury Leader of the House of Commons Whig Palmerston I 1857 7003111000000000000 ♠ 3 years and 14 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 20 February 1858 11 June 1859 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli II 7002477000000000000 ♠ 1 year and 112 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 12 June 1859 18 October 1865 1859 First Lord of the Treasury Leader of the House of Commons Liberal Palmerston II 1865 7003232100000000000 ♠ 6 years and 129 days The Right Honourable John Russell 1st Earl Russell KG PC FRS ( 1792 -- 1878 ) 29 October 1865 26 June 1866 -- First Lord of the Treasury Leader of the House of Lords Liberal Russell II 7002241000000000000 ♠ 241 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby KG PC ( 1799 -- 1869 ) 28 June 1866 25 February 1868 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli III 7002608000000000000 ♠ 1 year and 243 days The Right Honourable Benjamin Disraeli MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 -- 7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and 77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April 1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone II 23 April 1880 9 June 1885 1880 7003187400000000000 ♠ 5 years and 48 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 23 June 1885 28 January 1886 -- Sec. of State for Foreign Affairs Leader of the House of Lords Conservative Salisbury I 7002220000000000000 ♠ 220 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone III 1 February 1886 20 July 1886 1885 7002170000000000000 ♠ 170 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 July 1886 11 August 1892 1886 Leader of the House of Lords First Lord of the Treasury Sec. of State for Foreign Affairs Conservative Salisbury II 7003221000000000000 ♠ 6 years and 18 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone IV 15 August 1892 2 March 1894 1892 7002565000000000000 ♠ 1 year and 200 days The Right Honourable Archibald Primrose 5th Earl of Rosebery KG PC FRS ( 1847 -- 1929 ) 5 March 1894 22 June 1895 -- First Lord of the Treasury Lord President of the Council Leader of the House of Lords Liberal Rosebery 7002475000000000000 ♠ 1 year and 110 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 June 1895 11 July 1902 Leader of the House of Lords Sec. of State for Foreign Affairs Lord Keeper of the Privy Seal Conservative Salisbury III ( Con. -- Lib. U . ) 1895 1900 7003257300000000000 ♠ 7 years and 17 days Edward VII ( 1901 -- 1910 ) The Right Honourable Arthur Balfour FRS DL MP for Manchester East ( 1848 -- 1930 ) 11 July 1902 5 December 1905 -- First Lord of the Treasury Leader of the House of Commons Conservative Balfour 7003124400000000000 ♠ 3 years and 148 days The Right Honourable Sir Henry Campbell - Bannerman GCB MP for Stirling Burghs ( 1836 -- 1908 ) 5 December 1905 7 April 1908 First Lord of the Treasury Leader of the House of Commons Liberal Campbell - Bannerman 1906 7002855000000000000 ♠ 2 years and 125 days The Right Honourable Herbert Henry Asquith KC FRS MP for East Fife ( 1852 -- 1928 ) 7 April 1908 5 December 1916 First Lord of the Treasury Leader of the House of Commons Sec. of State for War Liberal Asquith I 1910 ​ 1910 George V ( 1910 -- 1936 ) ​ Asquith II ( Lib. -- Con. -- et al . ) 7003316500000000000 ♠ 8 years and 243 days The Right Honourable David Lloyd George OM MP for Caernarvon Boroughs ( 1863 -- 1945 ) 6 December 1916 19 October 1922 First Lord of the Treasury Liberal Lloyd George War 1918 Lloyd George II ( Lib. -- Con . ) 7003214400000000000 ♠ 5 years and 318 days The Right Honourable Bonar Law MP for Glasgow Central ( 1858 -- 1923 ) 23 October 1922 20 May 1923 1922 First Lord of the Treasury Leader of the House of Commons Conservative ( Scot. U . ) Law 7002210000000000000 ♠ 210 days The Right Honourable Stanley Baldwin JP MP for Bewdley ( 1867 -- 1947 ) 23 May 1923 16 January 1924 -- First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Conservative Baldwin I 7002239000000000000 ♠ 239 days The Right Honourable Ramsay MacDonald MP for Aberavon ( 1866 -- 1937 ) 22 January 1924 4 November 1924 1923 First Lord of the Treasury Leader of the House of Commons Sec. of State for Foreign Affairs Labour MacDonald I 7002288000000000000 ♠ 288 days The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 4 November 1924 5 June 1929 1924 First Lord of the Treasury Leader of the House of Commons Conservative Baldwin II 7003167500000000000 ♠ 4 years and 214 days The Right Honourable Ramsay MacDonald FRS MP for Seaham ( 1866 -- 1937 ) 5 June 1929 7 June 1935 1929 First Lord of the Treasury Leader of the House of Commons Labour MacDonald II National Labour National I ( N. Lab. -- Con. -- et al . ) 1931 National II 7003219400000000000 ♠ 6 years and 3 days ​ The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 7 June 1935 28 May 1937 1935 First Lord of the Treasury Leader of the House of Commons Conservative National III ( Con. -- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​ Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26 July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker ( Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26 July 1945 26 October 1951 1945 First Lord of the Treasury Minister of Defence Labour Attlee I 1950 Attlee II 7003228400000000000 ♠ 6 years and 93 days The Right Honourable Sir Winston Churchill KG OM CH TD PCc DL FRS RA MP for Woodford ( 1874 -- 1965 ) 26 October 1951 6 April 1955 1951 First Lord of the Treasury Minister of Defence Conservative Churchill III 7003125900000000000 ♠ 3 years and 163 days Elizabeth II ( 1952 -- present ) The Right Honourable Sir Anthony Eden KG MC MP for Warwick & Leamington ( 1897 -- 1977 ) 6 April 1955 10 January 1957 1955 First Lord of the Treasury Conservative Eden 7002646000000000000 ♠ 1 year and 280 days The Right Honourable Harold Macmillan FRS MP for Bromley ( 1894 -- 1986 ) 10 January 1957 19 October 1963 1959 First Lord of the Treasury Conservative Macmillan I ​ Macmillan II 7003247400000000000 ♠ 6 years and 283 days The Right Honourable Sir Alec Douglas - Home KT MP for Kinross & Western Perthshire ( 1903 -- 1995 ) 19 October 1963 16 October 1964 -- First Lord of the Treasury Conservative ( Scot. U . ) Douglas - Home 7002364000000000000 ♠ 364 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 16 October 1964 19 June 1970 1964 First Lord of the Treasury Minister for the Civil Service Labour Wilson I 1966 Wilson II 7003207300000000000 ♠ 5 years and 247 days The Right Honourable Edward Heath MBE MP for Bexley -- Sidcup ( 1916 -- 2005 ) 19 June 1970 4 March 1970 First Lord of the Treasury Minister for the Civil Service Conservative Heath 7003135500000000000 ♠ 3 years and 259 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 4 March 5 April 1976 First Lord of the Treasury Minister for the Civil Service Labour Wilson III Wilson IV 7002764000000000000 ♠ 2 years and 33 days The Right Honourable James Callaghan MP for Cardiff South East ( 1912 -- 2005 ) 5 April 1976 4 May 1979 -- First Lord of the Treasury Minister for the Civil Service Labour Callaghan 7003112500000000000 ♠ 3 years and 30 days The Right Honourable Margaret Thatcher FRS FRIC MP for Finchley ( 1925 -- 2013 ) See also § Premierships :3 First Lord of the Treasury Minister for the Civil Service Conservative Thatcher I 4 May 1979 28 November 1990 1979 Thatcher II Thatcher III 7003422700000000000 ♠ 11 years and 209 days The Right Honourable John Major MP for Huntingdon ( born 1943 ) 28 November 1990 2 May 1997 First Lord of the Treasury Minister for the Civil Service Conservative Major I ​ Major II 7003234800000000000 ♠ 6 years and 156 days The Right Honourable Tony Blair MP for Sedgefield ( born 1953 ) See also § Premierships : 4 First Lord of the Treasury Minister for the Civil Service Labour Blair I 2 May 1997 27 June 2007 1997 Blair II 2005 Blair III 7003370900000000000 ♠ 10 years and 57 days The Right Honourable Gordon Brown MP for Kirkcaldy & Cowdenbeath ( born 1951 ) See also § Premierships : 5 First Lord of the Treasury Minister for the Civil Service Labour Brown 27 June 2007 11 May -- 7003105000000000000 ♠ 2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July 2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days
  Fact 0.0: Robert Walpole, 1st Earl of Orford, was MP for King's Lynn and lived from 1676 to 1745.
  Fact 0.1: Robert Walpole was Prime Minister from 4 April 1721 to 11 February 1742, for 20 years and 314 days.
  Fact 0.2: Robert Walpole was a Whig.
  Fact 0.3: Spencer Compton, 1st Earl of Wilmington, lived from 1673 to 1743.
  Fact 0.4: Spencer Compton was Prime Minister from 16 February 1742 to 2 July 1743, for 1 year and 137 days.
  Fact 0.5: Spencer Compton was a Whig.
  Fact 0.6: Henry Pelham was MP for Sussex and lived from 1694 to 1754.
  Fact 0.7: Henry Pelham was Prime Minister from 27 August 1743 to 6 March 1754, for 10 years and 192 days.
  Fact 0.8: Henry Pelham was a Whig.
  Fact 0.9: Thomas Pelham-Holles, 1st Duke of Newcastle, lived from 1693 to 1768.
  Fact 0.10: Thomas Pelham-Holles was Prime Minister from 16 March 1754 to 16 November 1756, for 2 years and 246 days.
  Fact 0.11: Thomas Pelham-Holles was a Whig.
  Fact 0.12: William Cavendish, 4th Duke of Devonshire, lived from 1720 to 1764.
  Fact 0.13: William Cavendish was Prime Minister from 16 November 1756 to 25 June 1757, for 222 days.
  Fact 0.14: William Cavendish's ministry was a caretaker ministry in 1757.
  Fact 0.15: William Cavendish was a Whig.
  Fact 0.16: Thomas Pelham-Holles was Prime Minister again from 2 July 1757 to 26 May 1762, for 4 years and 329 days.
  Fact 0.17: Thomas Pelham-Holles was a Whig in his second term.
  Fact 0.18: John Stuart, 3rd Earl of Bute, lived from 1713 to 1792.
  Fact 0.19: John Stuart was Prime Minister from 26 May 1762 to 8 April 1763, for 318 days.
  Fact 0.20: John Stuart was a Tory.
  Fact 0.21: George Grenville was MP for Buckingham and lived from 1712 to 1770.
  Fact 0.22: George Grenville was Prime Minister from 16 April 1763 to 13 July 1765, for 2 years and 89 days.
  Fact 0.23: George Grenville was a Whig, in the Grenvillite faction.
  Fact 0.24: Charles Watson-Wentworth, 2nd Marquess of Rockingham, lived from 1730 to 1782.
  Fact 0.25: Charles Watson-Wentworth was Prime Minister from 13 July 1765 to 30 July 1766, for 1 year and 18 days.
  Fact 0.26: Charles Watson-Wentworth was a Whig, in the Rockinghamite faction.
  Fact 0.27: William Pitt, 1st Earl of Chatham, was MP for Bath and lived from 1708 to 1778.
  Fact 0.28: William Pitt, 1st Earl of Chatham, was Prime Minister from 30 July 1766 to 14 October 1768, for 2 years and 77 days.
  Fact 0.29: William Pitt, 1st Earl of Chatham, was a Whig, in the Chathamite faction.
  Fact 0.30: Augustus FitzRoy, 3rd Duke of Grafton, lived from 1735 to 1811.
  Fact 0.31: Augustus FitzRoy was Prime Minister from 14 October 1768 to 28 January 1770, for 1 year and 107 days.
  Fact 0.32: Augustus FitzRoy was a Whig, in the Chathamite faction.
  Fact 0.33: Frederick North, Lord North, was MP for Banbury and lived from 1732 to 1792.
  Fact 0.34: Frederick North was Prime Minister from 28 January 1770 to 22 March 1782, for 12 years and 54 days.
  Fact 0.35: Frederick North was a Tory, in the Northite faction.
  Fact 0.36: Charles Watson-Wentworth was Prime Minister again from 27 March 1782 to 1 July 1782, for 97 days.
  Fact 0.37: Charles Watson-Wentworth was a Whig in his second term, in the Rockinghamite faction.
  Fact 0.38: William Petty, 2nd Earl of Shelburne, lived from 1737 to 1805.
  Fact 0.39: William Petty was Prime Minister from 4 July 1782 to 2 April 1783, for 273 days.
  Fact 0.40: William Petty was a Whig, in the Chathamite faction.
  Fact 0.41: William Cavendish-Bentinck, 3rd Duke of Portland, lived from 1738 to 1809.
  Fact 0.42: William Cavendish-Bentinck was Prime Minister from 2 April 1783 to 19 December 1783, for 262 days.
  Fact 0.43: William Cavendish-Bentinck was a Whig.
  Fact 0.44: William Pitt the Younger was MP for Appleby and Cambridge University, and lived from 1759 to 1806.
  Fact 0.45: William Pitt the Younger was Prime Minister from 19 December 1783 to 14 March 1801, for 17 years and 86 days.
  Fact 0.46: William Pitt the Younger was a Tory, in the Pittite faction.
  Fact 0.47: Henry Addington was MP for Devizes and lived from 1757 to 1844.
  Fact 0.48: Henry Addington was Prime Minister from 17 March 1801 to 10 May 1804, for 3 years and 55 days.
  Fact 0.49: Henry Addington was a Tory, in the Pittite faction.
  Fact 0.50: William Pitt the Younger was Prime Minister again from 10 May 1804 to 23 January 1806, for 1 year and 259 days.
  Fact 0.51: William Pitt the Younger was a Tory, in the Pittite faction, in his second term.
  Fact 0.52: William Grenville, 1st Baron Grenville, lived from 1759 to 1834.
  Fact 0.53: William Grenville was Prime Minister from 11 February 1806 to 31 March 1807, for 1 year and 49 days.
  Fact 0.54: William Grenville was a Whig, in the All the Talents ministry.
  Fact 0.55: William Cavendish-Bentinck was Prime Minister again from 31 March 1807 to 4 October 1809, for 2 years and 188 days.
  Fact 0.56: William Cavendish-Bentinck was a Tory, in the Pittite faction, in his second term.
  Fact 0.57: Spencer Perceval was MP for Northampton and lived from 1762 to 1812.
  Fact 0.58: Spencer Perceval was Prime Minister from 4 October 1809 to 11 May 1812, for 2 years and 221 days.
  Fact 0.59: Spencer Perceval was a Tory, in the Pittite faction.
  Fact 0.60: Robert Jenkinson, 2nd Earl of Liverpool, lived from 1770 to 1828.
  Fact 0.61: Robert Jenkinson was Prime Minister from 8 June 1812 to 9 April 1827, for 14 years and 306 days.
  Fact 0.62: Robert Jenkinson was a Tory, in the Pittite faction.
  Fact 0.63: George Canning was MP for Seaford and lived from 1770 to 1827.
  Fact 0.64: George Canning was Prime Minister from 10 April 1827 to 8 August 1827, for 121 days.
  Fact 0.65: George Canning was a Tory, in the Canningite faction.
  Fact 0.66: Frederick John Robinson, 1st Viscount Goderich, lived from 1782 to 1859.
  Fact 0.67: Frederick John Robinson was Prime Minister from 31 August 1827 to 21 January 1828, for 144 days.
  Fact 0.68: Frederick John Robinson was a Tory, in the Canningite faction.
  Fact 0.69: Arthur Wellesley, 1st Duke of Wellington, lived from 1769 to 1852.
  Fact 0.70: Arthur Wellesley was Prime Minister from 22 January 1828 to 16 November 1830, for 2 years and 299 days.
  Fact 0.71: Arthur Wellesley was a Tory.
  Fact 0.72: Charles Grey, 2nd Earl Grey, lived from 1764 to 1845.
  Fact 0.73: Charles Grey was Prime Minister from 22 November 1830 to 9 July 1834, for 3 years and 230 days.
  Fact 0.74: Charles Grey was a Whig.
  Fact 0.75: William Lamb, 2nd Viscount Melbourne, lived from 1779 to 1848.
  Fact 0.76: William Lamb was Prime Minister from 16 July 1834 to 14 November 1834, for 122 days.
  Fact 0.77: William Lamb was a Whig.
  Fact 0.78: Arthur Wellesley was Prime Minister again from 14 November 1834 to 10 December 1834, for 27 days.
  Fact 0.79: Arthur Wellesley's second term was a caretaker ministry, and he was a Tory.
  Fact 0.80: Robert Peel was MP for Tamworth and lived from 1788 to 1850.
  Fact 0.81: Robert Peel was Prime Minister from 10 December 1834 to 8 April 1835, for 120 days.
  Fact 0.82: Robert Peel was a Conservative.
  Fact 0.83: William Lamb was Prime Minister again from 18 April 1835 to 30 August 1841, for 6 years and 135 days.
  Fact 0.84: William Lamb was a Whig in his second term.
  Fact 0.85: Robert Peel was Prime Minister again from 30 August 1841 to 29 June 1846, for 4 years and 304 days.
  Fact 0.86: Robert Peel was a Conservative in his second term.
  Fact 0.87: Lord John Russell was MP for the City of London and lived from 1792 to 1878.
  Fact 0.88: Lord John Russell was Prime Minister from 30 June 1846 to 21 February 1852, for 5 years and 237 days.
  Fact 0.89: Lord John Russell was a Whig.
  Fact 0.90: Edward Smith-Stanley, 14th Earl of Derby, lived from 1799 to 1869.
  Fact 0.91: Edward Smith-Stanley was Prime Minister from 23 February 1852 to 17 December 1852, for 299 days.
  Fact 0.92: Edward Smith-Stanley was a Conservative.
  Fact 0.93: George Hamilton-Gordon, 4th Earl of Aberdeen, lived from 1784 to 1860.
  Fact 0.94: George Hamilton-Gordon was Prime Minister from 19 December 1852 to 30 January 1855, for 2 years and 43 days.
  Fact 0.95: George Hamilton-Gordon was a Peelite.
  Fact 0.96: Henry John Temple, 3rd Viscount Palmerston, was MP for Tiverton and lived from 1784 to 1865.
  Fact 0.97: Henry John Temple was Prime Minister from 6 February 1855 to 19 February 1858, for 3 years and 14 days.
  Fact 0.98: Henry John Temple was a Whig.
  Fact 0.99: Edward Smith-Stanley was Prime Minister again from 20 February 1858 to 11 June 1859, for 1 year and 112 days.
  Fact 0.100: Edward Smith-Stanley was a Conservative in his second term.
  Fact 0.101: Henry John Temple was Prime Minister again from 12 June 1859 to 18 October 1865, for 6 years and 129 days.
  Fact 0.102: Henry John Temple was a Liberal in his second term.
  Fact 0.103: John Russell, 1st Earl Russell, lived from 1792 to 1878.
  Fact 0.104: John Russell, 1st Earl Russell, was Prime Minister from 29 October 1865 to 26 June 1866, for 241 days.
  Fact 0.105: John Russell, 1st Earl Russell, was a Liberal.
  Fact 0.106: Edward Smith-Stanley was Prime Minister again from 28 June 1866 to 25 February 1868, for 1 year and 243 days.
  Fact 0.107: Edward Smith-Stanley was a Conservative in his third term.
  Fact 0.108: Benjamin Disraeli was MP for Buckinghamshire and lived from 1804 to 1881.
  Fact 0.109: Benjamin Disraeli was Prime Minister from 27 February 1868 to 1 December 1868, for 279 days.
  Fact 0.110: Benjamin Disraeli was a Conservative.
  Fact 0.111: William Ewart Gladstone was MP for Greenwich and lived from 1809 to 1898.
  Fact 0.112: William Ewart Gladstone was Prime Minister from 3 December 1868 to 17 February 1874, for 5 years and 77 days.
  Fact 0.113: William Ewart Gladstone was a Liberal.
  Fact 0.114: Benjamin Disraeli, 1st Earl of Beaconsfield, was Prime Minister again from 20 February 1874 to 21 April 1880, for 6 years and 62 days.
  Fact 0.115: Benjamin Disraeli was a Conservative in his second term.
  Fact 0.116: William Ewart Gladstone was MP for Midlothian and was Prime Minister again from 23 April 1880 to 9 June 1885, for 5 years and 48 days.
  Fact 0.117: William Ewart Gladstone was a Liberal in his second term.
  Fact 0.118: Robert Gascoyne-Cecil, 3rd Marquess of Salisbury, lived from 1830 to 1903.
  Fact 0.119: Robert Gascoyne-Cecil was Prime Minister from 23 June 1885 to 28 January 1886, for 220 days.
  Fact 0.120: Robert Gascoyne-Cecil was a Conservative.
  Fact 0.121: William Ewart Gladstone was Prime Minister again from 1 February 1886 to 20 July 1886, for 170 days.
  Fact 0.122: William Ewart Gladstone was a Liberal in his third term.
  Fact 0.123: Robert Gascoyne-Cecil was Prime Minister again from 25 July 1886 to 11 August 1892, for 6 years and 18 days.
  Fact 0.124: Robert Gascoyne-Cecil was a Conservative in his second term.
  Fact 0.125: William Ewart Gladstone was Prime Minister again from 15 August 1892 to 2 March 1894, for 1 year and 200 days.
  Fact 0.126: William Ewart Gladstone was a Liberal in his fourth term.
  Fact 0.127: Archibald Primrose, 5th Earl of Rosebery, lived from 1847 to 1929.
  Fact 0.128: Archibald Primrose was Prime Minister from 5 March 1894 to 22 June 1895, for 1 year and 110 days.
  Fact 0.129: Archibald Primrose was a Liberal.
  Fact 0.130: Robert Gascoyne-Cecil was Prime Minister again from 25 June 1895 to 11 July 1902, for 7 years and 17 days.
  Fact 0.131: Robert Gascoyne-Cecil's third term was labelled Con. and Lib. U. in the table.
  Fact 0.132: Arthur Balfour was MP for Manchester East and lived from 1848 to 1930.
  Fact 0.133: Arthur Balfour was Prime Minister from 11 July 1902 to 5 December 1905, for 3 years and 148 days.
  Fact 0.134: Arthur Balfour was a Conservative.
  Fact 0.135: Henry Campbell-Bannerman was MP for Stirling Burghs and lived from 1836 to 1908.
  Fact 0.136: Henry Campbell-Bannerman was Prime Minister from 5 December 1905 to 7 April 1908, for 2 years and 125 days.
  Fact 0.137: Henry Campbell-Bannerman was a Liberal.
  Fact 0.138: Herbert Henry Asquith was MP for East Fife and lived from 1852 to 1928.
  Fact 0.139: Herbert Henry Asquith was Prime Minister from 7 April 1908 to 5 December 1916, for 8 years and 243 days.
  Fact 0.140: Herbert Henry Asquith was a Liberal.
  Fact 0.141: David Lloyd George was MP for Caernarvon Boroughs and lived from 1863 to 1945.
  Fact 0.142: David Lloyd George was Prime Minister from 6 December 1916 to 19 October 1922, for 5 years and 318 days.
  Fact 0.143: David Lloyd George was a Liberal.
  Fact 0.144: Bonar Law was MP for Glasgow Central and lived from 1858 to 1923.
  Fact 0.145: Bonar Law was Prime Minister from 23 October 1922 to 20 May 1923, for 210 days.
  Fact 0.146: Bonar Law was a Conservative.
  Fact 0.147: Stanley Baldwin was MP for Bewdley and lived from 1867 to 1947.
  Fact 0.148: Stanley Baldwin was Prime Minister from 23 May 1923 to 16 January 1924, for 239 days.
  Fact 0.149: Stanley Baldwin was a Conservative.
  Fact 0.150: Ramsay MacDonald was MP for Aberavon and lived from 1866 to 1937.
  Fact 0.151: Ramsay MacDonald was Prime Minister from 22 January 1924 to 4 November 1924, for 288 days.
  Fact 0.152: Ramsay MacDonald was a Labour politician.
  Fact 0.153: Stanley Baldwin was Prime Minister again from 4 November 1924 to 5 June 1929, for 4 years and 214 days.
  Fact 0.154: Stanley Baldwin was a Conservative in his second term.
  Fact 0.155: Ramsay MacDonald was MP for Seaham and was Prime Minister again from 5 June 1929 to 7 June 1935, for 6 years and 3 days.
  Fact 0.156: Ramsay MacDonald was a Labour politician in his second term.
  Fact 0.157: Stanley Baldwin was Prime Minister again from 7 June 1935 to 28 May 1937, for 1 year and 356 days.
  Fact 0.158: Stanley Baldwin was a Conservative in his third term.
  Fact 0.159: Neville Chamberlain was MP for Birmingham Edgbaston and lived from 1869 to 1940.
  Fact 0.160: Neville Chamberlain was Prime Minister from 28 May 1937 to 10 May 1940, for 2 years and 349 days.
  Fact 0.161: Neville Chamberlain was a Conservative.
  Fact 0.162: Winston Churchill was MP for Epping and lived from 1874 to 1965.
  Fact 0.163: Winston Churchill was Prime Minister from 10 May 1940 to 26 July 1945, for 5 years and 78 days.
  Fact 0.164: Winston Churchill was a Conservative.
  Fact 0.165: Clement Attlee was MP for Limehouse and Walthamstow West and lived from 1883 to 1967.
  Fact 0.166: Clement Attlee was Prime Minister from 26 July 1945 to 26 October 1951, for 6 years and 93 days.
  Fact 0.167: Clement Attlee was a Labour politician.
  Fact 0.168: Winston Churchill was MP for Woodford and was Prime Minister again from 26 October 1951 to 6 April 1955, for 3 years and 163 days.
  Fact 0.169: Winston Churchill was a Conservative in his second term.
  Fact 0.170: Anthony Eden was MP for Warwick and Leamington and lived from 1897 to 1977.
  Fact 0.171: Anthony Eden was Prime Minister from 6 April 1955 to 10 January 1957, for 1 year and 280 days.
  Fact 0.172: Anthony Eden was a Conservative.
  Fact 0.173: Harold Macmillan was MP for Bromley and lived from 1894 to 1986.
  Fact 0.174: Harold Macmillan was Prime Minister from 10 January 1957 to 19 October 1963, for 6 years and 283 days.
  Fact 0.175: Harold Macmillan was a Conservative.
  Fact 0.176: Alec Douglas-Home was MP for Kinross and Western Perthshire and lived from 1903 to 1995.
  Fact 0.177: Alec Douglas-Home was Prime Minister from 19 October 1963 to 16 October 1964, for 364 days.
  Fact 0.178: Alec Douglas-Home was a Conservative, in the Scottish Unionist faction.
  Fact 0.179: Harold Wilson was MP for Huyton and lived from 1916 to 1995.
  Fact 0.180: Harold Wilson was Prime Minister from 16 October 1964 to 19 June 1970, for 5 years and 247 days.
  Fact 0.181: Harold Wilson was a Labour politician.
  Fact 0.182: Edward Heath was MP for Bexley and Sidcup and lived from 1916 to 2005.
  Fact 0.183: Edward Heath was Prime Minister from 19 June 1970 to 4 March 1970, as the table lists it, for 3 years and 259 days.
  Fact 0.184: Edward Heath was a Conservative.
  Fact 0.185: Harold Wilson was Prime Minister again from 4 March to 5 April 1976, for 2 years and 33 days.
  Fact 0.186: Harold Wilson was a Labour politician in his second term.
  Fact 0.187: James Callaghan was MP for Cardiff South East and lived from 1912 to 2005.
  Fact 0.188: James Callaghan was Prime Minister from 5 April 1976 to 4 May 1979, for 3 years and 30 days.
  Fact 0.189: James Callaghan was a Labour politician.
  Fact 0.190: Margaret Thatcher was MP for Finchley and lived from 1925 to 2013.
  Fact 0.191: Margaret Thatcher was Prime Minister from 4 May 1979 to 28 November 1990, for 11 years and 209 days.
  Fact 0.192: Margaret Thatcher was a Conservative.
  Fact 0.193: John Major was MP for Huntingdon and was born in 1943.
  Fact 0.194: John Major was Prime Minister from 28 November 1990 to 2 May 1997, for 6 years and 156 days.
  Fact 0.195: John Major was a Conservative.
  Fact 0.196: Tony Blair was MP for Sedgefield and was born in 1953.
  Fact 0.197: Tony Blair was Prime Minister from 2 May 1997 to 27 June 2007, for 10 years and 57 days.
  Fact 0.198: Tony Blair was a Labour politician.
  Fact 0.199: Gordon Brown was MP for Kirkcaldy and Cowdenbeath and was born in 1951.
  Fact 0.200: Gordon Brown was Prime Minister from 27 June 2007 to 11 May (the year is not shown in the table), for 2 years and 319 days.
  Fact 0.201: Gordon Brown was a Labour politician.
  Fact 0.202: David Cameron was MP for Witney and was born in 1966.
  Fact 0.203: David Cameron was Prime Minister from 11 May (the year is not shown in the table) to 13 July 2016, for 6 years and 64 days.
  Fact 0.204: David Cameron was a Conservative, in a coalition with the Liberal Democrats.
  Fact 0.205: Theresa May was MP for Maidenhead and was born in 1956.
  Fact 0.206: Theresa May has been Prime Minister since 13 July 2016.
  Fact 0.207: Theresa May's term was 1 year and 193 days as of the table.
  Fact 0.208: Theresa May was a Conservative.
  Fact 0.209: The monarch Elizabeth II reigned from 1952 to the present.
  Fact 0.210: The monarch George I reigned from 1714 to 1727.
  Fact 0.211: The monarch George II reigned from 1727 to 1760.
  Fact 0.212: The monarch George III reigned from 1760 to 1820.
  Fact 0.213: The monarch George IV reigned from 1820 to 1830.
  Fact 0.214: The monarch William IV reigned from 1830 to 1837.
  Fact 0.215: The monarch Victoria reigned from 1837 to 1901.
  Fact 0.216: The monarch Edward VII reigned from 1901 to 1910.
  Fact 0.217: The monarch George V reigned from 1910 to 1936.
  Fact 0.218: The monarch Edward VIII reigned in 1936.
  Fact 0.219: The monarch George VI reigned from 1936 to 1952.

Retrieved items:
<item 1>
List of Prime Ministers of the United Kingdom
`` Boroughs to Lose One Member '' . Parliamentary Debates ( Hansard ) . 296 . House of Commons . col. 156 -- 157 . Jump up ^ Sir Donald Horne Macfarlane ( 14
April 1885 ) . `` Central Asia -- Russia and Afghanistan -- the Russo -- Afghan Frontier -- Russian Advances '' . Parliamentary Debates ( Hansard ) . 296 .
House of Commons . col. 1634 -- 1635 . ^ Jump up to : Marriott ( 1923 ) , p. 83 . Jump up ^ Clarke ( 1999 ) , p. 266 . Jump up ^ Hennessy ( 2001 ) , p. 39 .
Jump up ^ `` Parties and Prime Ministers '' . BBC News . 19 May 1998 . Archived from the original on 18 March 2004 . Retrieved 12 October 2008 . Jump up ^
Mackay , Robert ( 28 December 1987 ) . `` Thatcher longest serving British prime minister '' . United Press International .
</item 1>

<item 2>
List of Prime Ministers of the United Kingdom
British Prime Ministers . Amberley Publishing Limited . ISBN 978 - 1 - 4456 - 1242 - 3 . Quinault , Roland ( 17 March 2011 ) . British Prime Ministers and
Democracy : From Disraeli to Blair . Bloomsbury Publishing . ISBN 978 - 1 - 4411 - 1105 - 0 . External links ( edit ) `` Past Prime Ministers '' . gov.uk .
Government of the United Kingdom . Archived from the original on 25 August 2008 . `` Prime Ministers and Politics Timeline '' . BBC History . Archived from the
original on 27 May 2011 . ( hide ) Lists relating to Prime Ministers of the United Kingdom List of Prime Ministers of the United Kingdom Lists by order Age
Constituencies Cultural depictions Governments Historical rankings Number living Records Service ( timeline ) Term length Peerages By monarch Victoria Edward
VII George V Edward VIII George VI Elizabeth II Personal life Burial places Coats of arms Education Nickname Spouse In fiction Fictional Prime Ministers
Biography portal British politics portal Politics portal Lists portal Retrieved from `` https://en.wikipedia.
</item 2>

<item 3>
List of Prime Ministers of the United Kingdom
wikipedia.org/w/index.php?title=List_of_Prime_Ministers_of_the_United_Kingdom&oldid=821003615 '' Categories : Lists of Prime Ministers of the United Kingdom
Lists of government ministers of the United Kingdom Lists of prime ministers Prime Ministers of the United Kingdom Hidden categories : Wikipedia articles
incorporating a citation from the ODNB Wikipedia articles incorporating a citation from the 1911 Encyclopaedia Britannica with Wikisource reference Wikipedia
articles incorporating text from the 1911 Encyclopædia Britannica Articles incorporating Cite DNB template Pages using div col without cols and colwidth
parameters Featured lists Use dmy dates from September 2011 Talk Contents About Wikipedia Беларуская Čeština Cymraeg Dansk Deutsch Eesti Ελληνικά Español
Esperanto فارسی Français Frysk Gàidhlig 한국어 Հայերեն Hrvatski Ido Bahasa Indonesia Íslenska ქართული Latina Latviešu
</item 3>

<item 4>
List of Prime Ministers of the United Kingdom
BBC News . 13 July 2016 . Retrieved 26 July 2016 . Lay summary ( 13 July 2016 ) . Bibliography ( edit ) Burt , Llewellyn C. ( 1874 ) . A synoptical history of
England ( 2nd ed . ) . London : Lockwood . Carpenter , Clive , ed. ( 1992 ) . The Guinness UK Data Book . Guinness . ISBN 978 - 0 - 85112 - 522 - 0 . Chapman ,
Richard A. ( 31 January 2002 ) . `` History : from earliest times to the present day '' . The Treasury in Public Policy - Making . Routledge . ISBN 978 - 1 -
134 - 86426 - 3 . Clarke , John ( 1999 ) . `` House of Hanover '' . In Fraser , Antonia . The Lives of the Kings & Queens of England . University of California
Press ( published February 2000 ) .
</item 4>

<item 5>
List of Prime Ministers of the United Kingdom
Davidson , Jonathan ( 30 November 2010 ) . Downing Street Blues : A History of Depression and Other Mental Afflictions in British Prime Ministers . McFarland .
ISBN 978 - 0 - 7864 - 5793 - 9 . Grube , Dennis ( 11 October 2013 ) . Prime Ministers and Rhetorical Governance . Palgrave Macmillan UK . ISBN 978 - 1 - 137 -
31836 - 7 . King , Anthony Stephen , ed. ( 1985 ) . The British Prime Minister ( 2nd ed . ) . Duke University Press . ISBN 0 - 8223 - 0635 - 2 . Leonard , Dick
( 10 July 2008 ) . Nineteenth Century Premiers : Pitt to Rosebery . Palgrave Macmillan UK . ISBN 978 - 0 - 230 - 22725 - 5 . Parker , Robert J. ( 15 March 2011
) . British Prime Ministers . Amberley Publishing Limited .
</item 5>

<item 6>
List of Prime Ministers of the United Kingdom
-- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 --
7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the
Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and
77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord
of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April
1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William
</item 6>

<item 7>
List of Prime Ministers of the United Kingdom
List of Prime Ministers of the United Kingdom - wikipedia List of Prime Ministers of the United Kingdom Jump to : navigation , search Top left : Walpole became
the first Prime Minister of the Kingdom of Great Britain in 1721 . Top right : Pitt the Younger became the first Prime Minister of the United Kingdom of Great
Britain and Ireland in 1801 . Bottom left : Lloyd George was the last Prime Minister of the United Kingdom of Great Britain and Ireland as a result of the Anglo
- Irish Treaty in 1922 . Bottom right : Baldwin was Prime Minister when the country was renamed the United Kingdom of Great Britain and Northern Ireland in 1927
. The Prime Minister of the United Kingdom is the head of the Government of the United Kingdom , and chairs Cabinet meetings . There is no specific date when
the office of Prime Minister first appeared , as the role was not created but rather evolved over a period of time . The term was used in the House of Commons
in 1805 and it was certainly in parliamentary use by the 1880s , and in 1905 the post of Prime Minister was officially given recognition in the order of
precedence .
</item 7>

<item 8>
List of Prime Ministers of the United Kingdom
Committee Scotland Government Parliament Wales Government National Assembly Reserved matters Northern Ireland Executive Assembly Administrative geography ( show
) England Subdivisions of England Local government in England Scotland Subdivisions of Scotland Local government in Scotland Wales Local government in Wales
Northern Ireland Local government in Northern Ireland Elections ( show ) Parliament constituencies Political parties Last election Next election Referendums
Foreign relations ( show ) History Foreign policy UK and the United Nations Politics of the European Union Commonwealth of Nations United Kingdom portal Other
countries Atlas Downing Street Historical rankings of Prime Ministers of the United Kingdom List of Prime Ministers of the United Kingdom by tenure Living Prime
Ministers of the United Kingdom Records of Prime Ministers of the United Kingdom List of current heads of government in the United Kingdom and dependencies
Royal prerogative in the United Kingdom List of British governments List of votes of no confidence in British governments List of United Kingdom general
elections British Prime Ministers -- Wikipedia book Premierships ^ Premierships of Benjamin Disraeli ^ Premierships of William Ewart Gladstone ^ Premiership of
Margaret Thatcher ^ Premiership of Tony Blair ^ Premiership of Gordon Brown ^ Premiership of David Cameron ^ Premiership of Theresa May Notes ( edit ) Coalition
government / hung parliament .
</item 8>

<item 9>
List of Prime Ministers of the United Kingdom
After the succession of King George I in 1714 , the arrangement of a commission of Lords of the Treasury as opposed to a single Lord High Treasurer became
permanent . For the next three years , the government was headed by Lord Townshend , who was appointed Secretary of State for the Northern Department .
Subsequently , Lord Stanhope and Lord Sunderland ran the government jointly , with Stanhope managing foreign affairs and Sunderland managing domestic affairs .
Stanhope died in February 1721 and Sunderland resigned two months later ; Townshend and Robert Walpole were then invited to form the next government . From that
point , the holder of the office of First Lord also unofficially held the status of Prime Minister . It was not until the Edwardian era that the title Prime
Minister was constitutionally recognised . With only few exceptions , the Prime Minister still holds the office of First Lord by constitutional convention .
Since 1721 ( edit ) By century and monarch 18th to 19th century George I George II George III George IV William IV Victoria 20th to 21st century Edward VII
George V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative (
</item 9>

<item 10>
List of Prime Ministers of the United Kingdom
`` Thatcher longest serving British prime minister '' . United Press International . Retrieved 26 June 2017 . Jump up ^ Burt ( 1874 ) , p. 106 . Jump up ^
Seaman ( 2003 ) . Jump up ^ chpt. 4 , Royal and Parliamentary Titles Act 1927 , 12 April 1927 . Retrieved on 3 September 2017 . ^ Jump up to : Leonard ( 2010 )
, p. 1 . ^ Jump up to : Carpenter ( 1992 ) , p. 37 . Jump up ^ Leonard ( 2010 ) , p. 47 . ^ Jump up to : Leonard ( 2010 ) , p. 65 . Jump up ^ `` Prime ministers
of the United Kingdom ( 1730 -- 2011 ) '' . Oxford Dictionary of National Biography ( online ed . ) . Oxford University Press . ( Subscription or UK public
library membership required . ) ^ Jump up to : Chisholm , Hugh , ed. ( 1911 ) . `` Treasury '' .
</item 10>
=== END ITEM id=11494f8a6120a450 ===

=== ITEM id=34653f4e10fbf78c ===
Gold sentences and facts:
Sentence 0: 2016 production of still wine State Production ( gal ) Production ( % ) Alabama 34,966 0.004 % Arizona 190,008 0.024 % Arkansas 246,363 0.031 % California 680,272,512 84.354 % Colorado 556,994 0.069 % Connecticut 134,517 0.017 % Florida 1,634,103 0.203 % Georgia 276,144 0.034 % Idaho 497,007 0.062 % Illinois 391,805 0.049 % Indiana 1,411,540 0.175 % Iowa 335,522 0.042 % Kansas 104,129 0.013 % Kentucky 2,176,059 0.270 % Louisiana 38,232 0.005 % Maine 48,222 0.006 % Maryland 436,185 0.054 % Massachusetts 792,884 0.098 % Michigan 2,576,238 0.319 % Minnesota 331,946 0.041 % Missouri 993,831 0.123 % Montana 33,445 0.004 % Nebraska 120,366 0.015 % New Hampshire 159,316 0.020 % New Jersey 1,832,325 0.227 % New Mexico 749,818 0.093 % New York 27,969,308 3.468 % North Carolina 1,903,060 0.236 % Ohio 5,938,738 0.736 % Oklahoma 70,204 0.009 % Oregon 11,822,972 1.466 % Pennsylvania 12,405,181 1.538 % South Carolina 77,842 0.010 % South Dakota 139,738 0.017 % Tennessee 1,279,752 0.159 % Texas 1,907,299 0.237 % Vermont 2,172,526 0.269 % Virginia 2,157,395 0.268 % Washington 40,747,190 5.053 % West Virginia 40,733 0.005 % Wisconsin 1,129,405 0.140 % Others 312,051 0.039 % Sum 806,447,891 100 %
  Fact 0.0: The table reports the 2016 production of still wine.
  Fact 0.1: Alabama produced 34,966 gallons of still wine in 2016, which is 0.004 percent of production.
  Fact 0.2: Arizona produced 190,008 gallons of still wine in 2016, which is 0.024 percent of production.
  Fact 0.3: Arkansas produced 246,363 gallons of still wine in 2016, which is 0.031 percent of production.
  Fact 0.4: California produced 680,272,512 gallons of still wine in 2016, which is 84.354 percent of production.
  Fact 0.5: Colorado produced 556,994 gallons of still wine in 2016, which is 0.069 percent of production.
  Fact 0.6: Connecticut produced 134,517 gallons of still wine in 2016, which is 0.017 percent of production.
  Fact 0.7: Florida produced 1,634,103 gallons of still wine in 2016, which is 0.203 percent of production.
  Fact 0.8: Georgia produced 276,144 gallons of still wine in 2016, which is 0.034 percent of production.
  Fact 0.9: Idaho produced 497,007 gallons of still wine in 2016, which is 0.062 percent of production.
  Fact 0.10: Illinois produced 391,805 gallons of still wine in 2016, which is 0.049 percent of production.
  Fact 0.11: Indiana produced 1,411,540 gallons of still wine in 2016, which is 0.175 percent of production.
  Fact 0.12: Iowa produced 335,522 gallons of still wine in 2016, which is 0.042 percent of production.
  Fact 0.13: Kansas produced 104,129 gallons of still wine in 2016, which is 0.013 percent of production.
  Fact 0.14: Kentucky produced 2,176,059 gallons of still wine in 2016, which is 0.270 percent of production.
  Fact 0.15: Louisiana produced 38,232 gallons of still wine in 2016, which is 0.005 percent of production.
  Fact 0.16: Maine produced 48,222 gallons of still wine in 2016, which is 0.006 percent of production.
  Fact 0.17: Maryland produced 436,185 gallons of still wine in 2016, which is 0.054 percent of production.
  Fact 0.18: Massachusetts produced 792,884 gallons of still wine in 2016, which is 0.098 percent of production.
  Fact 0.19: Michigan produced 2,576,238 gallons of still wine in 2016, which is 0.319 percent of production.
  Fact 0.20: Minnesota produced 331,946 gallons of still wine in 2016, which is 0.041 percent of production.
  Fact 0.21: Missouri produced 993,831 gallons of still wine in 2016, which is 0.123 percent of production.
  Fact 0.22: Montana produced 33,445 gallons of still wine in 2016, which is 0.004 percent of production.
  Fact 0.23: Nebraska produced 120,366 gallons of still wine in 2016, which is 0.015 percent of production.
  Fact 0.24: New Hampshire produced 159,316 gallons of still wine in 2016, which is 0.020 percent of production.
  Fact 0.25: New Jersey produced 1,832,325 gallons of still wine in 2016, which is 0.227 percent of production.
  Fact 0.26: New Mexico produced 749,818 gallons of still wine in 2016, which is 0.093 percent of production.
  Fact 0.27: New York produced 27,969,308 gallons of still wine in 2016, which is 3.468 percent of production.
  Fact 0.28: North Carolina produced 1,903,060 gallons of still wine in 2016, which is 0.236 percent of production.
  Fact 0.29: Ohio produced 5,938,738 gallons of still wine in 2016, which is 0.736 percent of production.
  Fact 0.30: Oklahoma produced 70,204 gallons of still wine in 2016, which is 0.009 percent of production.
  Fact 0.31: Oregon produced 11,822,972 gallons of still wine in 2016, which is 1.466 percent of production.
  Fact 0.32: Pennsylvania produced 12,405,181 gallons of still wine in 2016, which is 1.538 percent of production.
  Fact 0.33: South Carolina produced 77,842 gallons of still wine in 2016, which is 0.010 percent of production.
  Fact 0.34: South Dakota produced 139,738 gallons of still wine in 2016, which is 0.017 percent of production.
  Fact 0.35: Tennessee produced 1,279,752 gallons of still wine in 2016, which is 0.159 percent of production.
  Fact 0.36: Texas produced 1,907,299 gallons of still wine in 2016, which is 0.237 percent of production.
  Fact 0.37: Vermont produced 2,172,526 gallons of still wine in 2016, which is 0.269 percent of production.
  Fact 0.38: Virginia produced 2,157,395 gallons of still wine in 2016, which is 0.268 percent of production.
  Fact 0.39: Washington produced 40,747,190 gallons of still wine in 2016, which is 5.053 percent of production.
  Fact 0.40: West Virginia produced 40,733 gallons of still wine in 2016, which is 0.005 percent of production.
  Fact 0.41: Wisconsin produced 1,129,405 gallons of still wine in 2016, which is 0.140 percent of production.
  Fact 0.42: Other states produced 312,051 gallons of still wine in 2016, which is 0.039 percent of production.
  Fact 0.43: The total 2016 production of still wine was 806,447,891 gallons, which is 100 percent.

Retrieved items:
<item 1>
American wine
Although a Vitis vinifera variety , it is a grape of `` very modest '' quality . Jean - Louis Vignes was one of the early settlers to use a higher quality
vinifera in his vineyard near Los Angeles . The first winery in the United States to become commercially successful was founded in Cincinnati , Ohio in the
mid-1830s by Nicholas Longworth . He made a sparkling wine from Catawba grapes . By 1855 Ohio had 1500 acres in vineyards , according to travel writer Frederick
Law Olmsted , who said it was more than in Missouri and Illinois , which each had 1100 acres in wine . German immigrants from the late 1840s had been
instrumental in building the wine industry in those states . In the 1860s , vineyards in the Ohio River Valley were attacked by Black rot . This prompted
several winemakers to move north to the Finger Lakes region of western New York . During this time , the Missouri wine industry , centered on the German colony
in Hermann , was expanding rapidly along both shores of the Missouri River west of St. Louis .
</item 1>

<item 2>
American wine
American wine - wikipedia American wine Jump to : navigation , search American wine has been produced for over 300 years . Today , wine production is undertaken
in all fifty states , with California producing 89 percent of all US wine . The United States is the fourth - largest wine producing country in the world after
France , Italy , and Spain . The North American continent is home to several native species of grape , including Vitis labrusca , Vitis riparia , Vitis
rotundifolia , and Vitis vulpina . But the wine making industry is based on the cultivation of the European Vitis vinifera , which was introduced by European
settlers . With more than 1,100,000 acres ( 4,500 km ) under vine , the United States is the sixth-most planted country in the world after France , Italy ,
Spain , China and Turkey . Contents ( hide ) 1 History 2 Wine regions 2.1 Production by state 3 Appellation system 3.1 Appellation labeling laws 3.2
Semi-generic wines 3.3 Fighting varietals 4 Other U.S.
</item 2>

<item 3>
American wine
During the Great Depression , consumers demanded cheap `` jug wine '' ( so - called dago red ) and sweet , fortified ( high alcohol ) wine . Before Prohibition
dry table wines outsold sweet wines by three to one , but afterward , the ratio of demand changed dramatically . As a result , by 1935 , 81 % of California 's
production was sweet wines . For decades , wine production was low and limited . Leading the way to new methods of wine production was research conducted at the
University of California , Davis and some of the state universities in New York . Faculty at the universities published reports on which varieties of grapes
grew best in which regions , held seminars on winemaking techniques , consulted with grape growers and winemakers , offered academic degrees in viticulture ,
and promoted the production of quality wines . In the 1970s and 1980s , success by Californian winemakers in the northern part of the state helped to secure
foreign investment from other winemaking regions , most notably the Champenois of France . Winemakers also cultivated vineyards in Oregon and Washington , on
Long Island in New York , and numerous other new locales .
</item 3>

<item 4>
American wine
labeling laws 5 Three - tier distribution 6 Largest producers 7 See also 8 References 9 Further reading 10 External links History ( edit ) Main article :
History of American wine Sparkling wine made from a second label of Pleasant Valley Wine Company in New York which was the first bonded winery in the United
States . The first Europeans to explore North America , a Viking expedition from Greenland , called it Vinland because of the profusion of grape vines they
found . The earliest wine made in what is now the United States was produced between 1562 and 1564 by French Huguenot settlers from Scuppernong grapes at a
settlement near Jacksonville , Florida . In the early American colonies of Virginia and the Carolinas , wine making was an official goal laid out in the
founding charters . However , settlers discovered that the wine made from the various native grapes had flavors which were unfamiliar and which they did not
like . This led to repeated efforts to grow the familiar European Vitis vinifera varieties , beginning with the Virginia Company exporting French vinifera vines
with French vignerons to Virginia in 1619 .
</item 4>

<item 5>
American wine
Prior to the early 1970s , all grapes had to be from the vintage year . All labels must list the alcohol content based on percentage by volume . For bottles
labeled as varietals at least 75 % of the grape must be of the variety . In Oregon , the requirement is 90 % . American wine labels are also required to list if
they contain sulfites and carry the Surgeon General 's warning about alcohol consumption . Three - tier distribution ( edit ) Following the repeal of
Prohibition , the United States federal government allowed each state to regulate its own production and sale of alcohol . For the majority of states this led
to the development of a three - tier distribution system between the producer , wholesaler and consumer . Depending on the state there are some exceptions ,
with wineries allowed to sell directly to consumers on site at the winery or to ship wine across state lines . Some states allow interstate sales through
e-commerce . In the 2005 case of Granholm v. Heald , the Supreme Court of the United States struck down state laws banning interstate shipments but allowing in
- state sales .
</item 5>

<item 6>
American wine
Americans became more educated about wines and increased their demand for high - quality wine . All 50 states now have some acreage in vineyard cultivation . By
2004 , 668 million gallons ( 25.3 million hectoliters ) of wine were consumed in the United States . Today , the U.S. produces over 800 million gallons of wine
a year , of which California produces more than 84 % , followed by Washington , New York , Pennsylvania , and Oregon . In the second decade of the 21st century
, the US wine industry faces the growing challenges of competition from international exports and managing domestic regulations on interstate sales and shipment
of wine . Wine regions ( edit ) The Pacific Northwest includes in the winemaking regions of Washington State where varieties like Cabernet Sauvignon are grown .
There are nearly 3,000 commercial vineyards in the United States , and at least one winery in each of the 50 states . West Coast -- More than 90 % of the total
American wine production occurs in the states of California , Washington and Oregon .
</item 6>

<item 7>
American wine
wine sales and is the second largest producer in the world . Constellation Brands - With foreign wine holdings Constellation is the largest producer in the
world and includes Robert Mondavi Winery and Columbia Winery in its portfolio The Wine Group - San Francisco - based business which owns the Franzia box wine
label , Concannon Vineyard and Mogen David kosher wine . Bronco Wine Company - Owners of the Charles Shaw wine `` Two Buck Chuck '' line which accounts for
nearly 5 million of Bronco 's annual average 9 million cases per year . Diageo - UK based company with American holdings in Sterling Vineyards , Beaulieu
Vineyard and Chalone Vineyard Brown - Forman Corporation - Owners of the Korbel Champagne Cellars brand Beringer Blass - Australian based wine division of
Foster 's Group and owner of the Beringer wine and Stags ' Leap Winery brands Jackson Wine Estates - Owners of the Kendall - Jackson brand See also ( edit )
Wine portal American Viticultural Area Globalization of wine Judgment of Paris References ( edit ) Jump up ^ United States Department of Agriculture `` Global
Wine Report August 2006 Archived April 8 , 2008 , at the Wayback Machine.
</item 7>

<item 8>
American wine
522 0.042 % Kansas 104,129 0.013 % Kentucky 2,176,059 0.270 % Louisiana 38,232 0.005 % Maine 48,222 0.006 % Maryland 436,185 0.054 % Massachusetts 792,884 0.098
% Michigan 2,576,238 0.319 % Minnesota 331,946 0.041 % Missouri 993,831 0.123 % Montana 33,445 0.004 % Nebraska 120,366 0.015 % New Hampshire 159,316 0.020 %
New Jersey 1,832,325 0.227 % New Mexico 749,818 0.093 % New York 27,969,308 3.468 % North Carolina 1,903,060 0.236 % Ohio 5,938,738 0.736 % Oklahoma 70,204
0.009 % Oregon 11,822,972 1.466 % Pennsylvania 12,405,181 1.538 % South Carolina 77,842 0.010 % South Dakota 139,738 0.
</item 8>

<item 9>
American wine
This Supreme Court decision meant that states could decide to allow both out - of - state wine sales and in - state sales , or ban both together . Convenience
stores and retail stores are large distributors of wine , with over 175,000 outlets that sell wine across the United States . In addition , there are around
332,000 other locations ( bars , restaurants , etc . ) that sell wine , contributing to the $30 + billion in annual sales over the past three years . In 2010 ,
the average monthly per - store sales of wine jumped to nearly $12,000 from $9,084 in 2009 . The average gross margin dollars from wine increased to $3,324 from
$2,616 in the year prior , with gross margin percentages up to an average 28.2 percent in 2010 , versus 27 percent in 2009 . Largest producers ( edit ) As of
2005 The largest producers of American wine . E & J Gallo Winery - Accounts for more than a quarter of all U.S. wine sales and is the second largest producer in
the world .
</item 9>

<item 10>
American wine
Rocky Mountain Region -- Notably Idaho and Colorado Southwestern United States -- Notably Texas and New Mexico Midwestern United States -- Notably Missouri and
Illinois Great Lakes region -- Notably Michigan , northern New York and Ohio East Coast of the United States -- Notably western New York State and eastern Long
Island , New Jersey , Pennsylvania , Virginia , North Carolina and Florida . Production by state ( edit ) Production of still wine per state in 2016 was as
follows : 2016 production of still wine State Production ( gal ) Production ( % ) Alabama 34,966 0.004 % Arizona 190,008 0.024 % Arkansas 246,363 0.031 %
California 680,272,512 84.354 % Colorado 556,994 0.069 % Connecticut 134,517 0.017 % Florida 1,634,103 0.203 % Georgia 276,144 0.034 % Idaho 497,007 0.062 %
Illinois 391,805 0.049 % Indiana 1,411,540 0.175 % Iowa 335,522 0.042 % Kansas 104,129 0.013 % Kentucky 2,
</item 10>
=== END ITEM id=34653f4e10fbf78c ===

=== ITEM id=b004ab7a6895bec2 ===
Gold sentences and facts:
Sentence 0: Take action immediately upon hearing thunder . Anyone close enough to the storm to hear thunder can be struck by lightning . Avoid electrical appliances , including corded telephones . Cordless and wireless telephones are safe to use during a thunderstorm . Close and stay away from windows and doors , as glass can become a serious hazard in high wind . Do not bathe or shower , as plumbing conducts electricity . If driving , safely exit the roadway , turn on hazard lights , and park . Remain in the vehicle and avoid touching metal .
  Fact 0.0: Take action immediately upon hearing thunder.
  Fact 0.1: Anyone close enough to the storm to hear thunder can be struck by lightning.
  Fact 0.2: Avoid electrical appliances, including corded telephones.
  Fact 0.3: Cordless and wireless telephones are safe to use during a thunderstorm.
  Fact 0.4: Close and stay away from windows and doors.
  Fact 0.5: Glass can become a serious hazard in high wind.
  Fact 0.6: Do not bathe or shower.
  Fact 0.7: Plumbing conducts electricity.
  Fact 0.8: If driving, safely exit the roadway.
  Fact 0.9: If driving, turn on hazard lights.
  Fact 0.10: If driving, park.
  Fact 0.11: Remain in the vehicle and avoid touching metal.
Sentence 1: Take action immediately upon hearing thunder . Anyone close enough to the storm to hear thunder can be struck by lightning .
  Fact 1.0: Take action immediately upon hearing thunder.
  Fact 1.1: Anyone close enough to the storm to hear thunder can be struck by lightning.

Retrieved items:
<item 1>
Severe thunderstorms can occur year-round and kill or injure people despite warnings; a similar event struck Boscastle, England, on 16 August 2004.
Cloud-to-ground lightning can ignite wildfires, especially during low-precipitation storms when vegetation is dry, and causes fatalities, including among
outdoor workers in lightning-prone Florida. Lightning-related nitric oxide can contribute to acid rain, which damages infrastructure, soils, plants, and aquatic
and terrestrial life. Thunderstorms producing hail that reaches
</item 1>

<item 2>
Thunderstorm
Radio pulses produced by cosmic rays are being used to study how electric charges develop within thunderstorms . More organized meteorological projects such as
VORTEX2 use an array of sensors , such as the Doppler on Wheels , vehicles with mounted automated weather stations , weather balloons , and unmanned aircraft to
investigate thunderstorms expected to produce severe weather . Lightning is detected remotely using sensors that detect cloud - to - ground lightning strokes
with 95 percent accuracy in detection and within 250 metres ( 820 ft ) of their point of origin . Mythology and religion ( edit ) Thunderstorms strongly
influenced many early civilizations . Greeks believed that they were battles waged by Zeus , who hurled lightning bolts forged by Hephaestus . Some American
Indian tribes associated thunderstorms with the Thunderbird , who they believed was a servant of the Great Spirit . The Norse considered thunderstorms to occur
when Thor went to fight Jötnar , with the thunder and lightning being the effect of his strikes with the hammer Mjölnir . Hinduism recognizes Indra as the god
of rain and thunderstorms .
</item 2>

<item 3>
A thunderstorm, also called an electrical storm, is a storm associated with a cumulonimbus cloud and characterized by lightning and thunder. The source includes
references on severe storms and cloud physics, plus links to storm anatomy and meteorological concepts such as buoyancy, wind shear, condensation, and
visibility.
</item 3>

<item 4>
Supercells are the strongest thunderstorms; wind-shear-driven tropical and subtropical storm systems can help form hurricanes, while dry lightning can ignite
wildfires. Researchers study storms with radar, weather stations, photography, lightning sensors, and coordinated projects using mobile Doppler radar, balloons,
and aircraft. Lightning sensors detect cloud-to-ground strikes with 95% accuracy, within 250 metres. Thunderstorms inspired diverse religious myths, and
lightning has also been observed—or is suspected—on Jupiter, Saturn, Neptune
</item 4>

<item 5>
Thunderstorm
doi : 10.1111 / cea. 12709 . PMID 26765082 ^ Jump up to : American Red Cross . `` Thunderstorm Safety Checklist '' ( PDF ) . American Red Cross . Retrieved 24
August 2011 . ^ Jump up to : National Weather Service Weather Forecast Office . `` Thunderstorm '' . Severe Weather Preparedness Information . Albuquerque , NM
: NOAA . Retrieved 24 August 2011 . Jump up ^ Federal Emergency Management Agency . `` Thunderstorms and Lightning '' . Ready . US Department of Homeland
Security . Archived from the original on 23 June 2011 . Retrieved 24 August 2011 . ^ Jump up to : Federal Emergency Management Agency . `` What to Do Before a
Thunderstorm '' . US Department of Homeland Security . Archived from the original on 20 August 2011 . Retrieved 24 August 2011 . ^ Jump up to : `` NWS Lightning
Safety Myths '' . Lightningsafety.noaa.gov. 2014 - 06 - 30 .
</item 5>

<item 6>
The references cover thunderstorm and lightning safety guidance from the Red Cross, NOAA, and FEMA; lightning myths and FAQs; thunderstorm frequency and
tropical-cyclone hazards; weather analysis and severe-storm operations; and NASA/Fermi findings on antimatter-producing thunderstorms. A clinical article is
also cited (doi:10.1111/cea.12709; PMID 26765082).
</item 6>

<item 7>
Thunderstorm
The National Weather Service ( NWS ) in the United States recommends several precautions that people should take if thunderstorms are likely to occur : Know the
names of local counties , cities , and towns , as these are how warnings are described . Monitor forecasts and weather conditions and know whether thunderstorms
are likely in the area . Be alert for natural signs of an approaching storm . Cancel or reschedule outdoor events ( to avoid being caught outdoors when a storm
hits ) . Take action early so you have time to get to a safe place . Get inside a substantial building or hard - topped metal vehicle before threatening weather
arrives . If you hear thunder , get to the safe place immediately . Avoid open areas like hilltops , fields , and beaches , and do n't be or be near the tallest
objects in an area when thunderstorms are occurring . Do n't shelter under tall or isolated trees during thunderstorms . If in the woods , put as much distance
between you and any trees during thunderstorms . If in a group , spread out so that you increase the chances for survivors who could come to the aid of any
victims from a lightning strike .
</item 7>

<item 8>
Thunderstorm
Safety ( edit ) While safety and preparedness often overlap , `` thunderstorm safety '' generally refers to what people should do during and after a storm . The
American Red Cross recommends that people follow these precautions if a storm is imminent or in progress : Take action immediately upon hearing thunder . Anyone
close enough to the storm to hear thunder can be struck by lightning . Avoid electrical appliances , including corded telephones . Cordless and wireless
telephones are safe to use during a thunderstorm . Close and stay away from windows and doors , as glass can become a serious hazard in high wind . Do not bathe
or shower , as plumbing conducts electricity . If driving , safely exit the roadway , turn on hazard lights , and park . Remain in the vehicle and avoid
touching metal . The NWS stopped recommending the `` lightning crouch '' in 2008 as it does n't provide a significant level of protection and will not
significantly lower the risk of being killed or injured from a nearby lightning strike . Frequent occurrences ( edit ) See also : United States rainfall
climatology A mild thunderstorm over Niagara Falls , Ontario .
</item 8>

<item 9>
Thunderstorm
Lightningsafety.noaa.gov. 2014 - 06 - 30 . Retrieved 2014 - 08 - 20 . Jump up ^ `` NWS JetStream -- Lightning Frequently Asked Questions '' . Srh.noaa.gov. 2014
- 06 - 28 . Retrieved 2014 - 08 - 20 . Jump up ^ `` You 're not safer crouching : Six things you did n't know about lightning '' . LA Times . Retrieved 2014 -
08 - 20 . Jump up ^ National Geographic Almanac of Geography , ISBN 0 - 7922 - 3877 - X , page 75 . Jump up ^ `` How many thunderstorms occur each year ? '' .
Thunderstorms . Sky Fire Productions . Archived from the original on 2007 - 07 - 11 . Retrieved 2006 - 08 - 23 . Jump up ^ National Weather Service JetStream (
2008 - 10 - 08 ) . `` Tropical Cyclone Hazards '' . National Weather Service Southern Region Headquarters .
</item 9>

<item 10>
Thunderstorm
A similar event occurred in Boscastle , England , on 16 August 2004 . Hazards ( edit ) Each year , many people are killed or seriously injured by severe
thunderstorms despite the advance warning . While severe thunderstorms are most common in the spring and summer , they can occur at just about any time of the
year . Cloud - to - ground lightning ( edit ) See also : Lightning strike and Wildfire A return stroke , cloud - to - ground lightning strike during a
thunderstorm . Cloud - to - ground lightning frequently occurs within the phenomena of thunderstorms and have numerous hazards towards landscapes and
populations . One of the more significant hazards lightning can pose is the wildfires they are capable of igniting . Under a regime of low precipitation ( LP )
thunderstorms , where little precipitation is present , rainfall can not prevent fires from starting when vegetation is dry as lightning produces a concentrated
amount of extreme heat . Direct damage caused by lightning strikes occurs on occasion . In areas with a high frequency for cloud - to - ground lightning , like
Florida , lightning causes several fatalities per year , most commonly to people working outside . Acid rain is also a frequent risk produced by lightning .
</item 10>
=== END ITEM id=b004ab7a6895bec2 ===

=== ITEM id=716896be455f24ee ===
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
Most Thanksgiving turkeys are stuffed with a bread - based mixture and roasted . Sage is the traditional herb added to the stuffing , along with chopped celery
, carrots , and onions . Other ingredients , such as chopped chestnuts or other tree nuts , crumbled sausage or bacon , cranberries , raisins , or apples , may
be added to stuffing . If this mixture is prepared outside the bird , it may be known as dressing . Deep - fried turkey is rising in popularity due to its
shorter preparation time , but carries safety risks . The consumption of turkey on Thanksgiving is so ingrained in American culture that each year since 1947 ,
the National Turkey Federation has presented a live turkey to the President of the United States prior to each Thanksgiving . These turkeys were initially
slaughtered and eaten for the President 's Thanksgiving dinner ; since 1989 , the presented turkeys have typically been given a mock pardon to great fanfare and
sent to a park to live out the rest of their usually short natural lives . Alternatives to turkey ( edit ) Non-traditional foods other than turkey are sometimes
served as the main dish for a Thanksgiving dinner . Ham is often served alongside turkey in many non-traditional households .
</item 1>

<item 2>
Thanksgiving dinner
Thanksgiving dinner - wikipedia Thanksgiving dinner Jump to : navigation , search ( Almost Traditional ) Thanksgiving Dinner The centerpiece of contemporary
Thanksgiving in the United States and Canada is a large meal , generally centered on a large roasted turkey . It is served with a variety of side dishes which
vary from traditional dishes such as mashed potatoes , stuffing , and cranberry sauce , to ones that reflect regional or cultural heritage . The majority of the
dishes in the traditional American version of Thanksgiving dinner are made from foods native to the New World , as according to tradition the Pilgrims received
these foods , or learned how to grow them , from the Native Americans . Thanksgiving dinner is the largest eating event in the United States ; people eat more
on Thanksgiving than on any other day of the year .
</item 2>

<item 3>
Thanksgiving dinner
Alexander Hamilton proclaimed that no `` Citizen of the United States should refrain from turkey on Thanksgiving Day , '' and Benjamin Franklin had high regard
for the wild turkey as an American icon , but turkey was uncommon as Thanksgiving fare until after 1800 . By 1857 , turkey had become part of the traditional
dinner in New England . A Thanksgiving Day dinner served to the Civilian Conservation Corps in 1935 included : pickles , green olives , celery , roast turkey ,
oyster stew , cranberry sauce , giblet gravy , dressing , creamed asparagus tips , snowflake potatoes , baked carrots , hot rolls , fruit salad , mince meat pie
, fruit cake , candies , grapes , apples , clams , fish , and many other food harvests . French drip coffee , cigars and cigarettes . The White House Cook Book
, 1887 , by Mrs. F.L. Gillette , et al.
</item 3>

<item 4>
Thanksgiving dinner
( Pumpkin pie was forbidden because it contained eggs . ) The National Poultry and Egg Board furiously lobbied the President to cease promoting the plan ; it
culminated in a truce at the National Thanksgiving Turkey Presentation shortly before Thanksgiving . Turkey was no longer forbidden , but Eggless Thursdays
remained for the rest of the year , meaning no pumpkin pie was served at the White House dinner that year . Turkey ( edit ) Oven roasted turkey Turkey is the
most common main dish of a Thanksgiving dinner , to the point where Thanksgiving is sometimes colloquially called `` Turkey Day . '' In 2006 , American turkey
growers were expected to raise 270 million turkeys , to be processed into five billion pounds of turkey meat valued at almost $8 billion , with one third of all
turkey consumption occurring in the Thanksgiving - Christmas season , and a per capita consumption of almost 18 pounds ( 8.2 kg ) . The Broad Breasted White
turkey is particularly bred for Thanksgiving dinner and similar large feasts ; its large size ( specimens can grow to over 40 pounds ) and meat content make it
ideal for such situations , although the breed must be artificially bred and suffers from health problems due to its size .
</item 4>

<item 5>
Thanksgiving dinner
Contents ( hide ) 1 Historical menus 2 Turkey 3 Alternatives to turkey 4 Side dishes 5 Beverages 6 See also 7 References Historical menus ( edit ) 1943
Thanksgiving Day dinner menu from USS Wake Island ( CVE - 65 ) According to what traditionally is known as `` The First Thanksgiving , '' the 1621 feast between
the Pilgrims and the Wampanoag at Plymouth Colony contained waterfowl , venison , ham , lobster , clams , berries , fruit , pumpkin , and squash . William
Bradford noted that , `` besides waterfowl , there was great store of wild turkeys , of which they took many . '' Many of the foods that were included in the
first feast ( except , notably , the seafood ) have since gone on to become staples of the modern Thanksgiving dinner . Early feasts of the Order of Good Cheer
, a French Canadian predecessor to the modern Thanksgiving , featured a potluck dinner with freshly - hunted fowl , game , and fish , hunted and shared by both
French Canadians and local natives . The use of the turkey in the US for Thanksgiving precedes Lincoln 's nationalization of the holiday in 1863 .
</item 5>

<item 6>
Thanksgiving dinner
There are also regional differences as to the stuffing or dressing traditionally served with the turkey . The traditional version has bread cubes , sage , onion
and celery . Southerners generally make their dressing from cornbread , while those in other parts of the country make stuffing from white , wheat or rye bread
as the base . One or several of the following may be added to the dressing / stuffing : oysters , apples , chestnuts , raisins , and sausages or the turkey 's
giblets . Other dishes reflect the region or cultural background of those who have come together for the meal . For example , Sauerkraut ( among those in the
Mid-Atlantic ; especially Baltimore ) is sometimes served . Many African Americans and Southerners serve baked macaroni and cheese and collard greens , along
with chitterlings and sweet potato pie , while some Italian - Americans often have lasagne on the table and Ashkenazi Jews may serve noodle kugel , a sweet
dessert pudding . Other Jewish families may consume foods commonly associated with Hanukkah , such as latkes or a sufganiyah ; the two holidays are usually in
close proximity and on extremely rare occasions overlap .
</item 6>

<item 7>
Thanksgiving dinner
In Alaskan villages , whale meat is sometimes eaten. Irish immigrants have been known to have prime rib of beef as their centerpiece since beef in Ireland was
once a rarity ; families would save up money for this dish to signify newfound prosperity and hope . In the United States , a globalist approach to Thanksgiving
has become common with the impact of immigration . Basic `` Thanksgiving '' ingredients , or the intent of the holiday , can be transformed to a variety of
dishes by using flavors , techniques , and traditions from their own cuisines . Others celebrate the holiday with a variety of dishes particularly when there is
a crowd to be fed , guest 's tastes vary and considering the financial means available . Side dishes ( edit ) A Thanksgiving meal in New England Green bean
casserole Many offerings are typically served alongside the main dish -- so many that , because of the amount of food , the Thanksgiving meal is sometimes
served midday or early afternoon to make time for all the eating , and preparation may begin at dawn or on days prior . Copious leftovers are also common
following the meal properly .
</item 7>

<item 8>
Thanksgiving dinner
It is not unheard of for Mexican Americans to serve their turkey with mole and roasted corn . In Puerto Rico , the Thanksgiving meal is completed with arroz con
gandules ( rice with pigeon peas ) or arroz con maiz ( rice with corn ) , pasteles ( root tamales ) stuffed with turkey , pumpkin - coconut crème caramel , corn
bread with longaniza , potato salad , roasted white sweet potatoes and Spanish sparkling hard cider . Turkey in Puerto Rico is stuffed with mofongo . Cuban -
Americans traditionally serve the turkey alongside a small roasted pork and include white rice and black beans or kidney beans . Vegetarians or vegans have been
known to serve alternative entree centerpieces such as a large vegetable pie or a stuffed and baked pumpkin or tofu substitutes . Many Midwesterners ( such as
Minnesotans ) of Norwegian or Scandinavian descent set the table with lefse . Beverages ( edit ) The beverages at Thanksgiving can vary as much as the side
dishes , often depending on who is present at the table and their tastes . Spirits or cocktails sometimes may be served before the main meal .
</item 8>

<item 9>
Thanksgiving dinner
Ham is often served alongside turkey in many non-traditional households . Goose and duck , foods which were traditional European centerpieces of Christmas
dinners before being displaced , are now sometimes served in place of the Thanksgiving turkey . Sometimes , fowl native to the region where the meal is taking
place is used ; for example , an article in Texas Monthly magazine suggested quail as the main dish for a Texan Thanksgiving feast . John Madden , who appeared
on television for the NFL Thanksgiving Day game from 1981 to 2001 , frequently advocated his fondness for the turducken , deboned turkey , duck and chicken
nested inside each other then cooked . In a few areas of the West Coast of the United States , Dungeness crab is common as an alternate main dish , as crab
season starts in early November . Similarly , Thanksgiving falls within deer hunting season in the Northeastern United States , which encourages the use of
venison as a centerpiece . Vegetarians or vegans may have a tofu , wheat gluten or lentil - based substitute ; or stuffed squash . In Alaskan villages , whale
meat is sometimes eaten.
</item 9>

<item 10>
Thanksgiving dinner
Copious leftovers are also common following the meal properly . Traditional Thanksgiving foods are sometimes specific to the day of riced potatoes , and
although some of the foods might be seen at any semi-formal meal in the United States , the meal often has something of a ritual or traditional quality . Many
Americans would say it is `` incomplete '' without cranberry sauce , stuffing or dressing , mashed potatoes and gravy , and brussels sprouts . Other commonly
served dishes include winter squash and sweet potatoes , the latter often prepared with sweeteners such as brown sugar , molasses , or marshmallows . Fresh ,
canned , or frozen corn is popular and green beans are frequently served ; in particular , green bean casserole , a product invented in 1955 by the Campbell
Soup Company to promote use of its cream of mushroom soup , has become a Thanksgiving standard . A fresh salad may be included , especially on the West Coast .
Bread rolls or biscuits and cornbread , especially in the South and parts of New England , are served . For dessert , various pies are usually served ,
particularly pumpkin pie , though apple pie , mincemeat pie , sweet potato pie , and pecan pie are often served as well .
</item 10>
=== END ITEM id=716896be455f24ee ===
