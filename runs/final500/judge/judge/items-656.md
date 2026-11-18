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

=== ITEM id=6b70fc218d7faf36 ===
Gold sentences and facts:
Sentence 0: The Sentencing Reform Act , part of the Comprehensive Crime Control Act of 1984 , was a U.S. federal statute intended to increase consistency in United States federal sentencing . It established the United States Sentencing Commission . It also abolished federal parole , except for persons convicted under federal law before November 1 , 1987 , persons convicted under District of Columbia law , `` transfer treaty '' inmates , persons who violated military law who are in federal civilian prisons , and persons who are defendants in state cases and who are under the U.S. Marshals Service Witness Protection Program .
  Fact 0.0: The Sentencing Reform Act was part of the Comprehensive Crime Control Act of 1984.
  Fact 0.1: The Sentencing Reform Act was a U.S. federal statute.
  Fact 0.2: The Sentencing Reform Act was intended to increase consistency in United States federal sentencing.
  Fact 0.3: The Sentencing Reform Act established the United States Sentencing Commission.
  Fact 0.4: The Sentencing Reform Act abolished federal parole.
  Fact 0.5: The abolition of federal parole did not apply to persons convicted under federal law before November 1, 1987.
  Fact 0.6: The abolition of federal parole did not apply to persons convicted under District of Columbia law.
  Fact 0.7: The abolition of federal parole did not apply to transfer treaty inmates.
  Fact 0.8: The abolition of federal parole did not apply to persons who violated military law who are in federal civilian prisons.
  Fact 0.9: The abolition of federal parole did not apply to persons who are defendants in state cases and who are under the U.S. Marshals Service Witness Protection Program.

Retrieved items:
<item 1>
California Highway Patrol
Following conviction , Gwaltney was subsequently sentenced to 90 years , being eligible for parole after serving at least 30 years . Gwaltney died in federal
custody from a heart attack . Craig Peyer ( edit ) A former officer from Poway , is serving a 25 - years - to - life prison sentence for the on - duty
strangulation and murder of 20 - year - old Cara Knott on December 27 , 1986 . After she was killed , Knott 's body was discarded over an abandoned highway
bridge . Peyer was convicted of first - degree murder in 1988 . Though he has maintained innocence ever since , Peyer has been denied parole consistently . In
2004 the parole board offered Peyer the chance to prove his alleged innocence by providing a DNA sample to compare against a drop of blood found on Cara 's shoe
, using modern DNA profiling . Peyer declined to provide a sample , nor explain why he declined to do so .
</item 1>

<item 2>
Sentencing Reform Act
This United States federal legislation article is a stub . You can help Wikipedia by expanding it . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Sentencing_Reform_Act&oldid=813388767 '' Categories : United States federal criminal legislation Penal system in the
United States United States sentencing law 1984 in law 1984 in the United States United States federal legislation stubs Hidden categories : Use dmy dates from
August 2011 All stub articles Talk About Wikipedia Add links This page was last edited on 3 December 2017 , at 12 : 21 . About Wikipedia
</item 2>

<item 3>
Whistleblower Protection Act
Legislative history Introduced in the Senate as S. 20 by Carl Levin ( D -- MI ) on January 25 , 1989 Passed the Senate on March 16 , 1989 ( 97 - 0 , Roll call
vote 24 , via Senate.gov ) Passed the House on March 21 , 1989 ( Agreed voice vote ) Signed into law by President George H.W. Bush on April 10 , 1989 The
Whistleblower Protection Act of 1989 , Pub. L. 101 - 12 as amended , is a United States federal law that protects federal whistleblowers who work for the
government and report agency misconduct . A federal agency violates the Whistleblower Protection Act if agency authorities take ( or threaten to take )
retaliatory personnel action against any employee or applicant because of disclosure of information by that employee or applicant . Whistleblowers may file
complaints that they believe reasonably evidences a violation of a law , rule or regulation ; gross mismanagement ; gross waste of funds ; an abuse of authority
; or a substantial and specific danger to public health or safety .
</item 3>

<item 4>
National Flood Insurance Program
The CBRA enacted a set of maps depicting the John H. Chafee Coastal Barrier Resources System ( CBRS ) in which federal flood insurance is unavailable for new or
significantly improved structures . The National Flood Insurance Reform Act of 1994 codified the Community Rating System ( an incentive program that encourages
communities to exceed the minimal federal requirements for development within floodplains ) within the NFIP . The program was further amended by the Flood
Insurance Reform Act of 2004 , with the goal of reducing `` losses to properties for which repetitive flood insurance claim payments have been made . '' The
Biggert -- Waters Flood Insurance Reform Act of 2012 ( Biggert - Waters ) modified the NFIP . At the conclusion of 2011 , as Congress passed Biggert - Waters ,
the NFIP cumulative debt was over $17 billion . A core principle of Biggert - Waters was to change the NFIP premiums to match actuarial risk - based premiums
that better reflected the expected losses and real risk of flooding . These changes included removing discounts to many policies which were being sold below
actual actuarial risk targets and eliminating `` grandfathering '' of older rates .
</item 4>

<item 5>
Capital punishment in Australia
Australian Federal Police . Retrieved 3 December 2017 . Jump up ^ Finlay , Lorraine ( March 2011 ) . `` Exporting the Death Penalty ? Reconciling International
Police Cooperation and the Abolition of the Death Penalty in Australia '' ( pdf ) . Sydney Law Review . 33 ( 1 ) : 95 -- 117 . ISSN 0082 - 0512 . Retrieved 3
December 2017 . Jump up ^ Traditional Aboriginal Law and Punishment Archived 6 March 2009 at the Wayback Machine . -- Part V -- Aboriginal Customary Law and the
Criminal Justice System , Law Reform Commission of Western Australia -- Aboriginal Customary Laws Discussion Paper Jump up ^ `` British Justice '' . Museum
Victoria Australia . Retrieved 14 March 2017 . Jump up ^ The Hanging of Ronald Ryan : 40 Years Later Archived 28 December 2007 at the Wayback Machine ...
Cosmos.bcst.yahoo.com ( 2009 - 08 - 27 ) . Retrieved on 2012 - 09 - 10 . Jump up ^ Biography -- Jean Lee -- Australian Dictionary of Biography .
</item 5>

<item 6>
Sentencing Reform Act
Sentencing Reform Act - Wikipedia Sentencing Reform Act Jump to : navigation , search The Sentencing Reform Act , part of the Comprehensive Crime Control Act of
1984 , was a U.S. federal statute intended to increase consistency in United States federal sentencing . It established the United States Sentencing Commission
. It also abolished federal parole , except for persons convicted under federal law before November 1 , 1987 , persons convicted under District of Columbia law
, `` transfer treaty '' inmates , persons who violated military law who are in federal civilian prisons , and persons who are defendants in state cases and who
are under the U.S. Marshals Service Witness Protection Program . The act was passed by large majorities in both houses of Congress . References ( edit ) Jump up
^ `` An Overview of the United States Sentencing Commission '' ( PDF ) . United States Sentencing Commission . United States Sentencing Commission . Retrieved 9
August 2011 . Jump up ^ Barry L. Johnson . `` Sentencing Reform Act ( 1984 ) '' . eNotes . Retrieved 9 August 2011 .
</item 6>

<item 7>
United States Bill of Rights
194 . ^ Jump up to : Krantz , Sheldon ( 1986 ) . `` Cruel and Unusual Punishment '' . Encyclopedia of the American Constitution . -- via HighBeam Research (
subscription required ) . Retrieved July 16 , 2013 . Jump up ^ `` U * X * L Encyclopedia of U.S. History '' . UXL Encyclopedia of American History . January 1 ,
2009 . Retrieved July 15 , 2013 . ^ Jump up to : Weisberg , Robert ( 1986 ) . `` Capital Punishment '' . Encyclopedia of the American Constitution . -- via
HighBeam Research ( subscription required ) . Retrieved July 16 , 2013 . Jump up ^ `` The Constitution of the United States of America : Analysis and
Interpretation , Centennial Edition , Interim Edition : Alalysis of Cases Decided by the Supreme Court of the United States to June 26 , 2013 '' ( PDF ) .
Washington , DC : U.S. Government Printing Office . 2013 . pp. 1738 -- 39 .
</item 7>

<item 8>
Capital punishment in Australia
The last sentence of death in Western Australia ( indeed anywhere in Australia ) was passed in 1984 , but the female killer ( Brenda Hodge ) in question had her
sentence commuted to imprisonment for life , as was customary by this stage . Capital punishment was formally removed from the statutes of the state with the
passage of the Acts Amendment ( Abolition of Capital Punishment ) Act 1984 . South Australia ( edit ) The Adelaide Gaol was the site of forty - four hangings ,
from Joseph Stagg on 18 November 1840 to Glen Sabre Valance , murderer and rapist , on 24 November 1964 and three executions also occurred at Mount Gambier Gaol
. Two Ngarrindjeri men were controversially executed by hanging along the Coorong on 22 August 1840 , after a drumhead court - martial conducted by Police
Commissioner O'Halloran on the orders of Governor George Gawler . The men were found to be guilty of murdering the twenty - five survivors of the shipwreck
Maria a few months before .
</item 8>

<item 9>
California Highway Patrol
Of the incident , Ronald Reagan , who was governor of California at the time , said the following words : `` If anything worthwhile comes of this tragedy , it
should be the realization by every citizen that often the only thing that stands between them and losing everything they hold dear ... is the man wearing a
badge . '' An emotionally charged follow - up investigation followed the incident , but eventually led to a complete revision of procedures during high - risk
and felony stops . Firearms procedures have also changed fundamentally due to this incident , and physical methods of arrest have been improved . The police
baton and pepper spray have been added to the officer 's arsenal , with more in ‐ depth training in their use . The 25th anniversary of the Newhall Incident was
observed on April 6 , 1995 , at the present Newhall Area office , where a brick memorial pays tribute to Officers James Pence ( 6885 ) , Roger Gore ( 6547 ) ,
Walt Frago ( 6520 ) and George Alleyn ( 6290 ) .
</item 9>

<item 10>
Sentencing Reform Act
eNotes . Retrieved 9 August 2011 . Jump up ^ `` FREQUENTLY ASKED QUESTIONS ABOUT THE LACK OF PAROLE FOR FEDERAL PRISONERS '' ( Archive ) . Families Against
Mandatory Minimums . p. 2 / 4 . Retrieved on December 13 , 2015 . Jump up ^ Stith , Kate ; Koh , Steve Y. ( 1993 ) . `` The Politics of Sentencing Reform : The
Legislative History of the Federal Sentencing Guidelines '' . Faculty Scholarship Series . Yale Law School . Retrieved 9 August 2011 . This United States
federal legislation article is a stub . You can help Wikipedia by expanding it .
</item 10>
=== END ITEM id=6b70fc218d7faf36 ===

=== ITEM id=46d4510938063c6f ===
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
Biological functions Location of eukaryote nuclear DNA within the chromosomes . DNA usually occurs as linear chromosomes in eukaryotes , and circular
chromosomes in prokaryotes . The set of chromosomes in a cell makes up its genome ; the human genome has approximately 3 billion base pairs of DNA arranged into
46 chromosomes . The information carried by DNA is held in the sequence of pieces of DNA called genes . Transmission of genetic information in genes is achieved
via complementary base pairing . For example , in transcription , when a cell uses the information in a gene , the DNA sequence is copied into a complementary
RNA sequence through the attraction between the DNA and the correct RNA nucleotides . Usually , this RNA copy is then used to make a matching protein sequence
in a process called translation , which depends on the same interaction between RNA nucleotides . In alternative fashion , a cell may simply copy its genetic
information in a process called DNA replication . The details of these functions are covered in other articles ; here the focus is on the interactions between
DNA and other molecules that mediate the function of the genome .
</item 1>

<item 2>
DNA
However , identification can be complicated if the scene is contaminated with DNA from several people . DNA profiling was developed in 1984 by British
geneticist Sir Alec Jeffreys , and first used in forensic science to convict Colin Pitchfork in the 1988 Enderby murders case . The development of forensic
science and the ability to now obtain genetic matching on minute samples of blood , skin , saliva , or hair has led to re-examining many cases . Evidence can
now be uncovered that was scientifically impossible at the time of the original examination . Combined with the removal of the double jeopardy law in some
places , this can allow cases to be reopened where prior trials have failed to produce sufficient evidence to convince a jury . People charged with serious
crimes may be required to provide a sample of DNA for matching purposes . The most obvious defense to DNA matches obtained forensically is to claim that
cross-contamination of evidence has occurred . This has resulted in meticulous strict handling procedures with new cases of serious crime . DNA profiling is
also used successfully to positively identify victims of mass casualty incidents , bodies or body parts in serious accidents , and individual victims in mass
war graves , via matching to family members .
</item 2>

<item 3>
DNA
Genes and genomes Further information : Cell nucleus , Chromatin , Chromosome , Gene , and Noncoding DNA Genomic DNA is tightly and orderly packed in the
process called DNA condensation , to fit the small available volumes of the cell . In eukaryotes , DNA is located in the cell nucleus , with small amounts in
mitochondria and chloroplasts . In prokaryotes , the DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid . The genetic
information in a genome is held within genes , and the complete set of this information in an organism is called its genotype . A gene is a unit of heredity and
is a region of DNA that influences a particular characteristic in an organism . Genes contain an open reading frame that can be transcribed , and regulatory
sequences such as promoters and enhancers , which control transcription of the open reading frame . In many species , only a small fraction of the total
sequence of the genome encodes protein . For example , only about 1.5 % of the human genome consists of protein - coding exons , with over 50 % of human DNA
consisting of non-coding repetitive sequences .
</item 3>

<item 4>
DNA
In this way , the base on the old strand dictates which base appears on the new strand , and the cell ends up with a perfect copy of its DNA . Extracellular
nucleic acids Naked extracellular DNA ( eDNA ) , most of it released by cell death , is nearly ubiquitous in the environment . Its concentration in soil may be
as high as 2 μg / L , and its concentration in natural aquatic environments may be as high at 88 μg / L. Various possible functions have been proposed for eDNA
: it may be involved in horizontal gene transfer ; it may provide nutrients ; and it may act as a buffer to recruit or titrate ions or antibiotics .
Extracellular DNA acts as a functional extracellular matrix component in the biofilms of several bacterial species . It may act as a recognition factor to
regulate the attachment and dispersal of specific cell types in the biofilm ; it may contribute to biofilm formation ; and it may contribute to the biofilm 's
physical strength and resistance to biological stress . Cell - free fetal DNA is found in the blood of the mother , and can be sequenced to determine a great
deal of information about the developing fetus .
</item 4>

<item 5>
The excerpt lists sources on DNA, including a PBS transcript and an article about DNA and the Cohanim tradition. Most references concern DNA as digital data
storage: a 2013 *Nature* paper on practical, high-capacity storage in synthesized DNA, related *Wall Street Journal* coverage and commentary, a 2016
presentation on programmable bacterial membranes, and a German patent for molecular DNA storage. It also cites Friedrich Miescher’s 1871 work.
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
The DNA sequence may be aligned with other DNA sequences to identify homologous sequences and locate the specific mutations that make them distinct . These
techniques , especially multiple sequence alignment , are used in studying phylogenetic relationships and protein function . Data sets representing entire
genomes ' worth of DNA sequences , such as those produced by the Human Genome Project , are difficult to use without the annotations that identify the locations
of genes and regulatory elements on each chromosome . Regions of DNA sequence that have the characteristic patterns associated with protein - or RNA - coding
genes can be identified by gene finding algorithms , which allow researchers to predict the presence of particular gene products and their possible functions in
an organism even before they have been isolated experimentally . Entire genomes may also be compared , which can shed light on the evolutionary history of
particular organism and permit the examination of complex evolutionary events . DNA nanotechnology The DNA structure at left ( schematic shown ) will self -
assemble into the structure visualized by atomic force microscopy at right . DNA nanotechnology is the field that seeks to design nanoscale structures using the
molecular recognition properties of DNA molecules . Image from Strong , 2004 .
</item 7>

<item 8>
Recombinant DNA combines sequences from different sources and can be introduced into organisms using plasmids or viral vectors to produce proteins, support
research, or improve agriculture. DNA profiling compares variable repetitive DNA regions to identify people from biological evidence; developed by Alec Jeffreys
in 1984, it is used in forensics, victim identification, and paternity testing. Mixed or contaminated samples complicate interpretation, requiring strict
handling. DNA tests can estimate parentage with high probability, including during pregnancy
</item 8>

<item 9>
DNA
Modern biology and biochemistry make intensive use of these techniques in recombinant DNA technology . Recombinant DNA is a man - made DNA sequence that has
been assembled from other DNA sequences . They can be transformed into organisms in the form of plasmids or in the appropriate format , by using a viral vector
. The genetically modified organisms produced can be used to produce products such as recombinant proteins , used in medical research , or be grown in
agriculture . DNA profiling Further information : DNA profiling Forensic scientists can use DNA in blood , semen , skin , saliva or hair found at a crime scene
to identify a matching DNA of an individual , such as a perpetrator . This process is formally termed DNA profiling , but may also be called `` genetic
fingerprinting '' . In DNA profiling , the lengths of variable sections of repetitive DNA , such as short tandem repeats and minisatellites , are compared
between people . This method is usually an extremely reliable technique for identifying a matching DNA . However , identification can be complicated if the
scene is contaminated with DNA from several people .
</item 9>

<item 10>
DNA
This can be used in studies ranging from ecological genetics to anthropology ; For example , DNA evidence is being used to try to identify the Ten Lost Tribes
of Israel . Information storage Main article : DNA digital data storage In a paper published in Nature in January 2013 , scientists from the European
Bioinformatics Institute and Agilent Technologies proposed a mechanism to use DNA 's ability to code information as a means of digital data storage . The group
was able to encode 739 kilobytes of data into DNA code , synthesize the actual DNA , then sequence the DNA and decode the information back to its original form
, with a reported 100 % accuracy . The encoded information consisted of text files and audio files . A prior experiment was published in August 2012 . It was
conducted by researchers at Harvard University , where the text of a 54,000 - word book was encoded in DNA . Moreover , in living cells , the storage can be
turned active by enzymes . Light - gated protein domains fused to DNA processing enzymes are suitable for that task in vitro . Fluorescent exonucleases can
transmit the output according to the nucleotide they have read .
</item 10>
=== END ITEM id=46d4510938063c6f ===

=== ITEM id=e33f44ffea01b7f7 ===
Gold sentences and facts:
Sentence 0: The 1836 article and other sources make it clear that this verse was not used soon after 1745 , and certainly before the song became accepted as the British national anthem in the 1780s and 1790s . It was included as an integral part of the song in the Oxford Book of Eighteenth Century Verse of 1926 , although erroneously referencing the `` fourth verse '' to the Gentleman 's Magazine article of 1745 .
  Fact 0.0: The 1836 article and other sources show that this verse was not used soon after 1745.
  Fact 0.1: The 1836 article and other sources show that this verse was certainly not used before the song became accepted as the British national anthem in the 1780s and 1790s.
Sentence 1: The French Marquise de Créquy wrote in her Souvenirs that the words `` Grand Dieu , sauvez le Roi ! '' were indeed composed by Jean - Baptiste Lully in gratitude for the survival by Louis XIV of an anal fistula operation . ( The surgical knife purpose - made for the occasion is on display in the Musée d'histoire de la médecine . ) Allegedly , Lully had set words by Marie de Brinon to music , and Créquy claimed that the tune was later plagiarised by Handel , and so adopted by the British as `` un de leurs air nationaux '' . ( The entire story might have been intended as a joke . ) Translated into Latin under the title Domine , Salvum Fac Regem , it became the French anthem until 1792 . After the Battle of Culloden , the Hanover dynasty supposedly adopted this melody as the British anthem .
  Fact 1.0: The verse was included as an integral part of the song in the Oxford Book of Eighteenth Century Verse of 1926.
  Fact 1.1: The Oxford Book of Eighteenth Century Verse of 1926 erroneously referenced the 'fourth verse' to the Gentleman's Magazine article of 1745.

Retrieved items:
<item 1>
God Save the Queen
Republic . Retrieved 11 November 2017 . External links ( edit ) Wikisource has original text related to this article : God Save the Queen Wikimedia Commons has
media related to God Save the Queen . National Anthem at the Royal Family website Streaming audio , lyrics and information about God Save the Queen Department
of Canadian Heritage -- Royal anthem page God Save Great George our King : -- article discussing different versions of the lyrics Free sheet music of God Save
the Queen from Cantorion.org A Point of View : Is it time for a new British national anthem ? BBC NEWS . Published 15 January 2016 . Retrieved 17 July 2017 .
National anthems of Europe Sovereign states Albania Andorra Armenia Austria Azerbaijan Belarus Belgium Bosnia and Herzegovina Bulgaria Croatia Cyprus Czech
Republic Denmark Estonia Finland France Georgia Germany Greece Hungary Iceland Ireland Italy Kazakhstan Latvia Liechtenstein Lithuania Luxembourg Macedonia
Malta Moldova Monaco Montenegro Netherlands Norway Poland Portugal Romania Russia San Marino Serbia Slovakia Slovenia Spain Sweden Switzerland Turkey Ukraine
United Kingdom Vatican City Territories , dependencies , other areas United Kingdom Constituent countries England Northern Ireland Scotland Wales British
Overseas Territories Gibraltar Crown dependencies Guernsey Isle of Man Jersey Counties Cornwall Lincolnshire Yorkshire Russia Adygea
</item 1>

<item 2>
God Save the Queen
National or royal anthem of Antigua and Barbuda ( royal ) Australia ( royal ) and its dependency of Norfolk Island ( national and royal ) The Bahamas ( royal )
Barbados ( royal ) Belize ( royal ) Canada ( royal ) Grenada ( royal ) Jamaica ( royal ) New Zealand ( national and royal ) and some of its dependencies ( Cook
Islands and Niue ( royal ) and Tokelau ( national and royal ) ) Saint Kitts and Nevis ( royal ) Mauritius ( royal ) 1968 - 1992 Solomon Islands ( royal ) Tuvalu
( royal ) United Kingdom ( national and royal ) , its Crown Dependencies and its overseas territories Also known as `` God Save the King '' ( when the monarch
is male ) Music Unknown composer Adopted September 1745 Audio sample `` God Save the Queen '' file help `` God Save the Queen '' ( alternatively `` God Save the
King '' , depending on the gender of the reigning monarch ) is the national or royal anthem in a number of Commonwealth realms , their territories , and the
British Crown dependencies . The author of the tune is unknown , and it may originate in plainchant ; but an attribution to the composer John Bull is sometimes
made .
</item 2>

<item 3>
God Save the Queen
Send him victorious Happy and glorious Long to reign over us God save the king ! Use in the United Kingdom ( edit ) See also : National anthem of England The
phrase `` God Save the King '' in use as a rallying cry to the support of the monarch and the UK 's forces `` God Save the Queen '' is the national anthem of
the United Kingdom of Great Britain and Northern Ireland . Like many aspects of British constitutional life , its official status derives from custom and use ,
not from Royal Proclamation or Act of Parliament . In general only one or two verses are sung , but on rare occasions three . The variation in the UK of the
lyrics to `` God Save the Queen '' is the oldest amongst those currently used , and forms the basis on which all other versions used throughout the Commonwealth
are formed ; though , again , the words have varied throughout these years . England has no official national anthem of its own ; `` God Save the Queen '' is
treated as the English national anthem when England is represented at sporting events ( though there are some exceptions to this rule , such as cricket where
Jerusalem is used ) .
</item 3>

<item 4>
God Save the Queen
In April 2007 there was an early day motion , number 1319 , to the British Parliament to propose that there should be a separate England anthem : `` That this
House ... believes that all English sporting associations should adopt an appropriate song that English sportsmen and women , and the English public , would
favour when competing as England '' . An amendment ( EDM 1319A3 ) was proposed by Evan Harris that the song `` should have a bit more oomph than God Save The
Queen and should also not involve God . '' For more information see also : National anthem of England National anthem of Scotland Hen Wlad Fy Nhadau Londonderry
Air Use in other Commonwealth countries ( edit ) `` God Save the King / Queen '' was exported around the world via the expansion of the British Empire , serving
as each country 's national anthem . Throughout the Empire 's evolution into the Commonwealth of Nations , the song declined in use in most states which became
independent .
</item 4>

<item 5>
God Save the Queen
The 1836 article and other sources make it clear that this verse was not used soon after 1745 , and certainly before the song became accepted as the British
national anthem in the 1780s and 1790s . It was included as an integral part of the song in the Oxford Book of Eighteenth Century Verse of 1926 , although
erroneously referencing the `` fourth verse '' to the Gentleman 's Magazine article of 1745 . On the opposing side , Jacobite beliefs were demonstrated in an
alternative verse used during the same period : God bless the prince , I pray , God bless the prince , I pray , Charlie I mean ; That Scotland we may see Freed
from vile Presbyt'ry , Both George and his Feckie , Ever so , Amen . In May 1800 , following an attempt to assassinate King George III at London 's Drury Lane
theatre , playwright Richard Sheridan immediately composed an additional verse , which was sung from the stage the same night : From every latent foe From the
assassins blow God save the King O'er him Thine arm extend For Britain 's sake defend Our father , king , and friend God save the King !
</item 5>

<item 6>
God Save the Queen
The first music proper , a performance of the National Anthem was programmed by Christopher Strachey on the Mark II Manchester Electronic Computer at same venue
, in 1951 . Later that year , short extracts of three pieces , the first being the National Anthem , were recorded there by a BBC outside broadcasting unit :
the other pieces being `` Ba Ba Black Sheep , and `` In the Mood '' . Researchers at the University of Canterbury , Christchurch restored the acetate master
disc in 2016 and the results may be heard on Soundcloud . Reception ( edit ) The philosopher and reformer Jeremy Bentham praised `` God Save the King '' in 1796
: `` the melody recommending itself by beauty to the most polished ears , and by its simplicity to the rudest ear . A song of this complexion , implanted by the
habit of half a century in the mass of popular sentiment , can not be refused a place in the inventory of the national blessings . '' Ludwig van Beethoven wrote
`` I have to show the English a little of what a blessing ' God Save the King ' is '' . Alex Marshall , the British author of Republic or Death !
</item 6>

<item 7>
God Save the Queen
3.2 Official peace version 2.1. 4 Historic alternative verses 2.2 Performance in the UK 2.3 Other British anthems 3 Use in other Commonwealth countries 3.1
Australia 3.2 Canada 3.2. 1 Lyrics in Canada 3.3 New Zealand 3.4 Rhodesia 4 Use elsewhere 5 Musical adaptations 5.1 Classical composers 5.2 Rock adaptations 5.3
In popular culture 5.4 Computer music 6 Reception 6.1 Calls for a new national anthem / anthems 7 Notes 8 External links History ( edit ) In The Oxford
Companion to Music , Percy Scholes points out the similarities to an early plainsong melody , although the rhythm is very distinctly that of a galliard , and he
gives examples of several such dance tunes that bear a striking resemblance to `` God Save the King / Queen '' . Scholes quotes a keyboard piece by John Bull (
1619 ) which has some similarities to the modern tune , depending on the placing of accidentals which at that time were unwritten in certain cases and left to
the discretion of the player ( see musica ficta ) .
</item 7>

<item 8>
God Save the Queen
Alex Marshall , the British author of Republic or Death ! : Travels in Search of National Anthems , called the anthem `` ludicrous '' . Calls for a New national
anthem / anthems ( edit ) There have been calls within the UK for a new national anthem , whether it be for the United Kingdom itself , Britain and / or England
( which all currently use God Save the Queen ) . There are many reasons people cite for wishing for a new national anthem , such as : from a non-religious
standpoint ( most people in the UK appear to no longer be religious ) , claims of God Save the Queen being long outdated and irrelevant in the 21st century ,
rejection of odes to promoting war and rejection of praising the monarchy from a republican perspective . A further reason is that England has no anthem of its
own for sporting contests and the like , whereas Scotland , Northern Ireland , and Wales do ; Flower of Scotland , Londonderry Air , and Land of My Fathers fill
this niche ( the former two on an unofficial basis ) , while England tends to use God Save the Queen exclusively and also unofficially . Notes ( edit ) ^ Jump
up to : `` National Anthem '' .
</item 8>

<item 9>
God Save the Queen
It is the national anthem of the United Kingdom and one of two national anthems used by New Zealand since 1977 , as well as for several of the UK 's territories
that have their own additional local anthem . It is also the royal anthem -- played specifically in the presence of the monarch -- of all the aforementioned
countries , as well as Australia ( since 1984 ) , Canada ( since 1980 ) , Barbados and Tuvalu . In countries not previously part of the British Empire , the
tune of `` God Save the Queen '' has provided the basis for various patriotic songs , though still generally connected with royal ceremony . In the United
States , the melody is used for the patriotic song `` My Country , ' Tis of Thee '' . The melody is also used for the national anthem of Liechtenstein , `` Oben
am jungen Rhein '' . Beyond its first verse , which is consistent , `` God Save the Queen / King '' has many historic and extant versions . Since its first
publication , different verses have been added and taken away and , even today , different publications include various selections of verses in various orders .
In general , only one verse is sung .
</item 9>

<item 10>
God Save the Queen
Since 1953 , the anthem is sometimes preceded by a fanfare composed by Gordon Jacob for the coronation of Queen Elizabeth II . Alternative British versions (
edit ) There have been several attempts to improve the song by rewriting the words . In the nineteenth century there was some lively debate about the national
anthem and , even then , verse two was considered to be slightly offensive . Notably , the question arose over the phrase `` scatter her enemies . '' Some
thought it placed better emphasis on the respective power of Parliament and the Crown to change `` her '' to `` our '' ; others pointed out that the theology
was somewhat dubious and substituted `` thine '' instead . Sydney G.R. Coles wrote a completely new version , as did Canon F.K. Harford . In 1836 , William
Edward Hickson wrote four alternative verses . The first , third , and fourth of these verses are appended to the National Anthem in the English Hymnal ( which
only includes verses one and three of the original lyrics ) .
</item 10>
=== END ITEM id=e33f44ffea01b7f7 ===

=== ITEM id=cd5cda3c114e1eea ===
Gold sentences and facts:
Sentence 0: `` Lord Have Mercy on the Working Man '' is a song written by Kostas and recorded by American country music singer Travis Tritt . It was released in August 1992 as the first of five singles from his third studio album , T-R-O-U-B-L-E . The song became Tritt 's tenth entry on the Billboard Hot Country Singles & Tracks ( now Hot Country Songs ) charts , where it peaked at number 5 .
  Fact 0.0: Lord Have Mercy on the Working Man is a song.
  Fact 0.1: Kostas wrote the song Lord Have Mercy on the Working Man.
  Fact 0.2: Travis Tritt, an American country music singer, recorded the song Lord Have Mercy on the Working Man.
  Fact 0.3: Lord Have Mercy on the Working Man was released in August 1992.
  Fact 0.4: Lord Have Mercy on the Working Man was the first of five singles from Travis Tritt's third studio album T-R-O-U-B-L-E.
  Fact 0.5: Lord Have Mercy on the Working Man became Travis Tritt's tenth entry on the Billboard Hot Country Singles & Tracks charts.
  Fact 0.6: The Billboard Hot Country Singles & Tracks chart is now called Hot Country Songs.
  Fact 0.7: Lord Have Mercy on the Working Man peaked at number 5 on the Billboard chart.
Sentence 1: `` Lord Have Mercy on the Working Man '' Single by Travis Tritt from the album T-R-O-U-B-L-E B - side `` Lord Have Mercy on the Working Man '' ( album version ) Released August 10 , 1992 Format CD single Genre Country Length 4 : 55 ( album version ) Label Warner Bros. Nashville Songwriter ( s ) Kostas Producer ( s ) Gregg Brown Travis Tritt singles chronology `` Nothing Short of Dying '' ( 1992 ) `` Lord Have Mercy on the Working Man '' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Nothing Short of Dying '' ( 1992 ) ' `` Lord Have Mercy on the Working Man '' ' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 )
  Fact 1.0: Lord Have Mercy on the Working Man is a single by Travis Tritt from the album T-R-O-U-B-L-E.
  Fact 1.1: The B-side of the single is Lord Have Mercy on the Working Man (album version).
  Fact 1.2: The single was released on August 10, 1992.
  Fact 1.3: The single format is CD single.
  Fact 1.4: The genre of the single is country.
  Fact 1.5: The album version of the song has a length of 4:55.
  Fact 1.6: The label of the single is Warner Bros. Nashville.
  Fact 1.7: Kostas is the songwriter of Lord Have Mercy on the Working Man.
  Fact 1.8: Gregg Brown is the producer of Lord Have Mercy on the Working Man.
  Fact 1.9: The Travis Tritt singles chronology lists Nothing Short of Dying (1992), Lord Have Mercy on the Working Man (1992), and Can I Trust You with My Heart (1992).
  Fact 1.10: The Travis Tritt singles chronology then repeats Nothing Short of Dying (1992), Lord Have Mercy on the Working Man (1992), and Can I Trust You with My Heart (1992).

Retrieved items:
<item 1>
Lord Have Mercy on the Working Man
( 1995 ) . Country Music Culture : From Hard Times to Heaven . Univ . Press of Mississippi . p. 233 . ISBN 0 - 87805 - 722 - 6 . Retrieved 28 August 2009 . Jump
up ^ `` Top RPM Country Tracks : Issue 1864 . '' RPM . Library and Archives Canada . November 28 , 1992 . Retrieved August 15 , 2013 . Jump up ^ `` Travis Tritt
Chart History ( Hot Country Songs ) '' . Billboard . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Travis Tritt singles Country Club ``
Country Club '' `` Help Me Hold On '' `` I 'm Gonna Be Somebody '' `` Put Some Drive in Your Country '' `` Drift Off to Dream '' It 's All About to Change ``
Here 's a Quarter ( Call Someone Who Cares ) '' `` Anymore '' `` The Whiskey Ai n't Workin ' '' ( With Marty Stuart ) `` Nothing Short of Dying '' T-R-O-U-B-L-E
`` Lord Have Mercy on the Working Man '' ``
</item 1>

<item 2>
Lord Have Mercy on the Working Man
Critical reception ( edit ) Teresa M. Walker , in her review for the Gainesville Sun , said that with the assistance from superstars on the final chorus , the
song `` should shoot up the charts . '' Dave Larsen of the Dayton Daily News cited it as one of the stronger tracks on the album , saying that the album ``
works best when Tritt sticks with the populist approach . '' Alanna Nash of Entertainment Weekly said of the song , `` Tritt finds a solid image for his laborer
's lament ( ... ) But the melody is so slight that he resorts to an acoustic arrangement that evokes Jimmie Rodgers and the young Roy Acuff . '' Deborah Evans
Price , of Billboard magazine gave the song a mixed review , saying that while the final chorus of song features famous country artists , it was too bad that
the entire song is n't as strong as the last verse . Music video ( edit ) Jack Cole directed the song 's music video .
</item 2>

<item 3>
Lord Have Mercy on the Working Man
Music video ( edit ) Jack Cole directed the song 's music video . In the book Country Music Culture : From Hard Times to Heaven , Curtis W. Ellison wrote that
the song 's music video `` confronted a litany of personal oppressions attributed to government policy '' that coincided with Bill Clinton 's presidential
campaign . Chart performance ( edit ) `` Lord Have Mercy on the Working Man '' spent 20 weeks on the Billboard Hot Country Singles & Tracks ( now Hot Country
Songs ) charts in 1992 , peaking at number 5 . The song also reached number 10 on the Canadian RPM Country Tracks charts . Chart ( 1992 ) Peak position Canada
Country Tracks ( RPM ) 10 US Hot Country Songs ( Billboard ) 5 References ( edit ) ^ Jump up to : T-R-O-U-B-L-E ( cassette insert ) . Travis Tritt . Warner
Bros. Records . 1992 . 45048 . Jump up ^ Hurst , Jack ( 13 August 1992 ) . `` Satellite show Travis Tritt to launch new album from Park West '' . Chicago
Tribune . Retrieved 28 August 2009 .
</item 3>

<item 4>
Lord Have Mercy on the Working Man
It was released in August 1992 as the first of five singles from his third studio album , T-R-O-U-B-L-E . The song became Tritt 's tenth entry on the Billboard
Hot Country Singles & Tracks ( now Hot Country Songs ) charts , where it peaked at number 5 . Contents ( hide ) 1 Content 2 Critical reception 3 Music video 4
Chart performance 5 References 6 External links Content ( edit ) `` Lord Have Mercy on the Working Man '' is a moderate up - tempo whose lyrics centralize on a
theme of economic injustice towards blue collar workers . The instrumentation features various forms of percussion from Sam Bacco , including crotales , wobble
board , spoons and a broom . Richard Bennett and Wendell Cox play guitar solos before the third verse , and Brooks & Dunn , T. Graham Brown , George Jones ,
Little Texas , Dana McVicker , Tanya Tucker and Porter Wagoner all sing background vocals on the final chorus .
</item 4>

<item 5>
Lord Have Mercy on the Working Man
Dying '' T-R-O-U-B-L-E `` Lord Have Mercy on the Working Man '' `` Can I Trust You with My Heart '' `` T-R-O-U-B-L-E '' `` Looking Out for Number One '' ``
Worth Every Mile '' Ten Feet Tall and Bulletproof `` Foolish Pride '' `` Ten Feet Tall and Bulletproof '' `` Between an Old Memory and Me '' `` Tell Me I Was
Dreaming '' Greatest Hits : From the Beginning `` Sometimes She Forgets '' `` Only You ( And You Alone ) '' The Restless Kind `` More Than You 'll Ever Know ''
`` Where Corn Do n't Grow '' `` She 's Going Home with Me '' `` Helping Me Get Over You '' ( With Lari White ) `` Still in Love with You '' No More Looking over
My Shoulder `` If I Lost You '' `` No More Looking Over My Shoulder '' `` Start the Car '' Down the Road I Go `` Best of Intentions '' `` It 's a Great Day to
Be Alive '' `` Love of a Woman '' `` Modern Day Bonnie and Clyde '' Strong Enough `` Strong Enough to Be Your Man '' `` Country Ai n't Country '' My Honky Tonk
History `` The Girl 's Gone Wild '' ``
</item 5>

<item 6>
Lord Have Mercy on the Working Man
Chicago Tribune . Retrieved 28 August 2009 . ^ Jump up to : Whitburn , Joel ( 2006 ) . The Billboard Book of Top 40 Country Hits ( 2 ed . ) . Billboard Books .
p. 353 . ISBN 0 - 8230 - 8291 - 1 . Jump up ^ Walker , Teresa M. ( 4 September 1992 ) . `` Travis Tritt 's rockin ' country '' . Gainesville Sun . Archived from
the original on 12 July 2012 . Retrieved 28 August 2009 . Jump up ^ Larsen , Dave ( 18 September 1992 ) . `` Recordings on review '' . Dayton Daily News .
Retrieved 28 August 2009 . Jump up ^ Nash , Alanna ( 4 September 1992 ) . `` T-R-O-U-B-L-E in mind '' . Entertainment Weekly . Retrieved 28 August 2009 . Jump
up ^ Billboard , August 15 , 1992 Jump up ^ Ellison , Curtis W.
</item 6>

<item 7>
Lord Have Mercy on the Working Man
On'ry and Mean '' Retrieved from `` https://en.wikipedia.org/w/index.php?title=Lord_Have_Mercy_on_the_Working_Man&oldid=789364103 '' Categories : 1992 singles
Travis Tritt songs Songs written by Kostas ( songwriter ) Warner Bros. Records singles 1992 songs Hidden categories : Articles with hAudio microformats
Singlechart usages for Canadacountry Singlechart called without artist Singlechart called without song Singlechart usages for Billboardcountrysongs Talk
Contents About Wikipedia Add links This page was last edited on 6 July 2017 , at 22 : 41 . Text is available under the Creative Commons Attribution - ShareAlike
License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the
Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Lord Have Mercy on the Working Man
Lord Have Mercy on the Working Man - wikipedia Lord Have Mercy on the Working Man Jump to : navigation , search `` Lord Have Mercy on the Working Man '' Single
by Travis Tritt from the album T-R-O-U-B-L-E B - side `` Lord Have Mercy on the Working Man '' ( album version ) Released August 10 , 1992 Format CD single
Genre Country Length 4 : 55 ( album version ) Label Warner Bros. Nashville Songwriter ( s ) Kostas Producer ( s ) Gregg Brown Travis Tritt singles chronology ``
Nothing Short of Dying '' ( 1992 ) `` Lord Have Mercy on the Working Man '' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Nothing Short of Dying '' (
1992 ) ' `` Lord Have Mercy on the Working Man '' ' ( 1992 ) `` Can I Trust You with My Heart '' ( 1992 ) `` Lord Have Mercy on the Working Man '' is a song
written by Kostas and recorded by American country music singer Travis Tritt .
</item 8>

<item 9>
Lord Have Mercy on the Working Man
Ai n't Country '' My Honky Tonk History `` The Girl 's Gone Wild '' `` What Say You '' ( with John Mellencamp ) `` I See Me '' The Storm `` You Never Take Me
Dancing '' `` Something Stronger Than Me '' The Calm After ... `` Sometimes Love Just Ai n't Enough '' ( With Tyler Reese ) Other songs `` This One 's Gonna
Hurt You ( For a Long , Long Time ) '' ( with Marty Stuart ) `` Take It Easy '' `` The Devil Comes Back to Georgia '' ( With Mark O'Connor , Charlie Daniels ,
Johnny Cash , and Marty Stuart ) `` Honky Tonkin 's What I Do Best '' ( With Marty Stuart ) `` Hope '' `` Here 's Your Sign ( Get the Picture ) '' ( With Bill
Engvall ) `` Same Old Train '' `` Out of Control Raging Fire '' ( with Patty Loveless ) `` Southern Boy '' ( with Charlie Daniels ) `` Bible Belt '' ( with
Little Feat ) `` Move It On Over '' `` Lonesome , On'ry and Mean '' Retrieved from `` https://en.wikipedia.org/w/index.
</item 9>

<item 10>
What a Friend We Have in Jesus
The tune was in the labor protest movement due to its wide popularity . `` Dump the Bosses Off Your Backs '' ( words by John Brill ) was published in the IWW ``
Little Red Songbook '' and has been recorded by dozens of artists including Utah Phillips , Anne Feeney , and Joe Glazer . A Norwegian version was used in the
film Trollhunter . Published versions ( edit ) Rise Up Singing , page 98 The Lutheran Hymnal , hymn # 457 References ( edit ) Wikisource has original text
related to this article : What a Friend We Have in Jesus Jump up ^ `` What a Friend We Have in Jesus '' . Retrieved 2007 - 05 - 03 . Jump up ^ Polack , W.G. (
1941 ) . Handbook to the Lutheran Hymnal . St. Louis : Concordia . p. 323 . Jump up ^ `` Washington Phillips discography '' . wirz.de . Retrieved August 24 ,
2015 . Jump up ^ George Washington Phillips : Jesus Is My Friend at AllMusic .
</item 10>
=== END ITEM id=cd5cda3c114e1eea ===

=== ITEM id=c72b141cf19bc8a4 ===
Gold sentences and facts:
Sentence 0: A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three vanishing points .
  Fact 0.0: A vanishing point is a point on the image plane of a perspective drawing.
  Fact 0.1: In a perspective drawing, the two-dimensional perspective projections, or drawings, of mutually parallel lines in three-dimensional space appear to converge at a vanishing point.
  Fact 0.2: When the set of parallel lines is perpendicular to a picture plane, the construction is known as one-point perspective.
  Fact 0.3: In one-point perspective, the vanishing point corresponds to the oculus, or "eye point", from which the image should be viewed for correct perspective geometry.
  Fact 0.4: Traditional linear drawings use objects with one to three sets of parallels.
  Fact 0.5: Traditional linear drawings define one to three vanishing points.

Retrieved items:
<item 1>
Vanishing point
She notes , in terms of projective geometry , the vanishing point is the image of the point at infinity associated with L , as the sightline from O through the
vanishing point is parallel to L . Vanishing line ( edit ) As a vanishing point originates in a line , so a vanishing line originates in a plane α that is not
parallel to the picture π . Given the eye point O , and β the plane parallel to α and lying on O , then the vanishing line of α is β ∩ π . For example , when α
is the ground plane and β is the horizon plane , then the vanishing line of α is the horizon line β ∩ π . Anderson notes , `` Only one particular vanishing line
occurs , often referred to as the `` horizon '' . To put it simply , the vanishing line of some plane , say α , is obtained by the intersection of the image
plane with another plane , say β , parallel to the plane of interest ( α ) , passing through the camera center . For different sets of lines parallel to this
plane α , their respective vanishing points will lie on this vanishing line .
</item 1>

<item 2>
Vanishing point
Vanishing point - wikipedia Vanishing point Jump to : navigation , search For other uses , see Vanishing point ( disambiguation ) . A vanishing point can be
seen at the far end of this railroad . A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective
projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a
picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the
image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three
vanishing points . Part of a series on Graphical projection Planar ( show ) Parallel projection Orthographic projection Multiview projection Axonometric
projection Isometric projection Oblique projection Perspective projection Curvilinear perspective Reverse perspective Views ( show ) 2.
</item 2>

<item 3>
Vanishing point
If we know the coordinates of one of these points , say v , and the direction of a straight line on the image plane , which passes through a second point , say
v , we can compute the coordinates of both v and v 3 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x ,
y , f ) , v ≡ ( x , y , f ) be the three corresponding vanishing points respectively . The orthocenter of the triangle with vertices in the three vanishing
points is the intersection of the optical axis and the image plane . Curvilinear and reverse perspective ( edit ) A curvilinear perspective is a drawing with
either 4 or 5 vanishing points . In 5 - point perspective the vanishing points are mapped into a circle with 4 vanishing points at the cardinal headings N , W ,
S , E and one at the circle 's origin . A reverse perspective is a drawing with vanishing points that are placed outside the painting with the illusion that
they are `` in front of '' the painting . Single point perspective projection . Double point perspective projection .
</item 3>

<item 4>
Vanishing point
This is one - point perspective . Similarly , when the image plane intersects two world - coordinate axes , lines parallel to those planes will meet at infinity
and form two vanishing points . This is called two - point perspective . In three - point perspective the image plane intersects the x , y , and z axes and
therefore lines parallel to these axes intersect , resulting in three different vanishing points . See also : Property 1 below Theorem ( edit ) The vanishing
point theorem is the principal theorem in the science of perspective . It says that the image in a picture plane π of a line L in space , not parallel to the
picture , is determined by its intersection with π and its vanishing point . Some authors have used the phrase , `` the image of a line includes its vanishing
point '' . Guidobaldo del Monte gave several verifications , and Humphry Ditton called the result the `` main and Great Proposition '' . Brook Taylor wrote the
first book in English on perspective in 1714 , which introduced the term `` vanishing point '' and was the first to fully explain the geometry of multipoint
perspective , and historian Kirsti Andersen compiled these observations .
</item 4>

<item 5>
Vanishing point
Using perspective pinhole projections , a point on L projected on the image plane will have coordinates defined as , x ′ = f x / z = f d − bz / az y ′ = f y / z
= f c / z This is the parametric representation of the image L ′ of the line L with z as the parameter . When z → − ∞ it stops at the point ( x ′ , y ′ ) = ( −
fb / a , 0 ) on the x ′ axis of the image plane . This is the vanishing point corresponding to all parallel lines with slope − b / a in the plane π . All
vanishing points associated with different lines with different slopes belonging to plane π will lie on the x ′ axis , which in this case is the horizon line .
2 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x , y , f ) , v ≡ ( x , y , f ) be the three
corresponding vanishing points respectively .
</item 5>

<item 6>
Vanishing point
Single point perspective projection . Double point perspective projection . Pietro Perugino 's use of perspective in the Delivery of the Keys fresco at the
Sistine Chapel ( 1481 -- 82 ) helped bring the Renaissance to Rome . Detection of vanishing points ( edit ) Several methods for vanishing point detection make
use of the line segments detected in images . Other techniques involve considering the intensity gradients of the image pixels directly . There are
significantly large numbers of vanishing points present in an image . Therefore , the aim is to detect the vanishing points that correspond to the principal
directions of a scene . This is generally achieved in two steps . The first step , called the accumulation step , as the name suggests , clusters the line
segments with the assumption that a cluster will share a common vanishing point . The next step finds the principal clusters present in the scene and therefore
it is called the search step . In the accumulation step , the image is mapped onto a bounded space called the accumulator space . The accumulator space is
partitioned into units called cells . Barnard assumed this space to be a Gaussian sphere centered on the optical center of the camera as an accumulator space .
</item 6>

<item 7>
Vanishing point
say D , will have the same vanishing point or converge at the same vanishing points . Mathematically , let q ≡ ( x , y , f ) be a point lying on the image plane
, where f is the focal length ( of the camera associated with the image ) , and let v ≡ ( x / h , y / h , f / h ) be the unit vector associated with q , where h
= √ x + y + f . If we consider a straight line in space S with the unit vector n ≡ ( n , n , n ) and its vanishing point v , the unit vector associated with v
is equal to n , assuming both are assumed to point towards the image plane . When the image plane is parallel to two world - coordinate axes , lines parallel to
the axis which is cut by this image plane will meet at infinity i.e. at the vanishing point . Lines parallel to the other two axes will not form vanishing
points as they are parallel to the image plane . This is one - point perspective .
</item 7>

<item 8>
Drawing
Typically this convergence point is somewhere along the horizon , as buildings are built level with the flat surface . When multiple structures are aligned with
each other , such as buildings along a street , the horizontal tops and bottoms of the structures typically converge at a vanishing point . When both the fronts
and sides of a building are drawn , then the parallel lines forming a side converge at a second point along the horizon ( which may be off the drawing paper . )
This is a two - point perspective . Converging the vertical lines to a third point above or below the horizon then produces a three - point perspective . Depth
can also be portrayed by several techniques in addition to the perspective approach above . Objects of similar size should appear ever smaller the further they
are from the viewer . Thus the back wheel of a cart appears slightly smaller than the front wheel . Depth can be portrayed through the use of texture . As the
texture of an object gets further away it becomes more compressed and busy , taking on an entirely different character than if it was close . Depth can also be
portrayed by reducing the contrast in more distant objects , and by making their colors less saturated .
</item 8>

<item 9>
Vanishing point
5 D Bird 's - eye view Cross section Cutaway drawing Exploded view drawing Fisheye lens Panorama Worm 's - eye view Zoom lens Topics ( show ) 3D projection
Anamorphosis Axonometry Computer graphics Computer - aided design Descriptive geometry Engineering drawing Map projection Picture plane Plans ( drawings )
Projection ( linear algebra ) Projection plane Projective geometry Stereoscopy Technical drawing Vanishing point Video game graphics Viewing frustum Contents (
hide ) 1 Vector notation 2 Theorem 3 Vanishing line 4 Properties of vanishing points 5 Curvilinear and reverse perspective 6 Detection of vanishing points 7
Applications of vanishing points 8 See also 9 References 10 External links Vector notation ( edit ) A 2D construction of perspective viewing , showing the
formation of a vanishing point The vanishing point may also be referred to as the `` direction point '' , as lines having the same directional vector , say D ,
will have the same vanishing point or converge at the same vanishing points .
</item 9>

<item 10>
Vanishing point
The horizon line is a theoretical line that represents the eye level of the observer . If the object is below the horizon line , its vanishing lines angle up to
the horizon line . If the object is above , they slope down . All vanishing lines end at the horizon line . Properties of vanishing points ( edit ) 1 .
Projections of two sets of parallel lines lying in some plane π appear to converge , i.e. the vanishing point associated with that pair , on a horizon line , or
vanishing line H formed by the intersection of the image plane with the plane parallel to π and passing through the pinhole . Proof : Consider the ground plane
π , as y = c which is , for the sake of simplicity , orthogonal to the image plane . Also , consider a line L that lies in the plane π , which is defined by the
equation ax + bz = d .
</item 10>
=== END ITEM id=c72b141cf19bc8a4 ===
