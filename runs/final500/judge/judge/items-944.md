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

=== ITEM id=d6363c98187f2249 ===
Gold sentences and facts:
Sentence 0: Roger Federer was the defending champion and successfully defended his title , defeating Marin Čilić in the final , 6 -- 2 , 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 . It was Federer 's 20th Grand Slam singles title and record - equalling sixth Australian Open men 's singles title ( tied with Roy Emerson and Novak Djokovic ) . With the win , Federer became the first male player to win at least six titles at two Grand Slam tournaments ( six at the Australian Open and eight at Wimbledon ) . Federer became the oldest man to win a Grand Slam singles title in the Open era since Ken Rosewall in 1972 . The time span between Federer 's first Grand slam glory at Wimbledon and this latest , almost 15 years , is an Open era record in the men 's singles field . This was also the 10th time that Federer has defended a Grand Slam title , with the previous time being at the 2008 US Open . Čilić became the first Croatian player to reach a singles final at the Australian Open .
  Fact 0.0: Roger Federer was the defending champion.
  Fact 0.1: Roger Federer successfully defended his title.
  Fact 0.2: Roger Federer defeated Marin Čilić in the final.
  Fact 0.3: The final score was 6-2, 6-7, 6-3, 3-6, 6-1.
  Fact 0.4: The win was Roger Federer's 20th Grand Slam singles title.
  Fact 0.5: The win was Roger Federer's record-equalling sixth Australian Open men's singles title.
  Fact 0.6: Roger Federer's sixth Australian Open men's singles title tied him with Roy Emerson and Novak Djokovic.
  Fact 0.7: With the win, Roger Federer became the first male player to win at least six titles at two Grand Slam tournaments.
  Fact 0.8: Roger Federer won six of those titles at the Australian Open.
  Fact 0.9: Roger Federer won eight of those titles at Wimbledon.
  Fact 0.10: Roger Federer became the oldest man to win a Grand Slam singles title in the Open era since Ken Rosewall won one in 1972.
  Fact 0.11: The time span between Roger Federer's first Grand Slam title at Wimbledon and his latest Grand Slam title is almost 15 years.
  Fact 0.12: That time span is an Open era record in men's singles.
  Fact 0.13: This was Roger Federer's 10th defense of a Grand Slam title.
  Fact 0.14: Roger Federer's previous Grand Slam title defense was at the 2008 US Open.
  Fact 0.15: Marin Čilić became the first Croatian player to reach a singles final at the Australian Open.
Sentence 1: Men 's Singles 2018 Australian Open Champion Roger Federer Runner - up Marin Čilić Final score 6 -- 2 , 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 Details Draw 128 ( 16 Q / 8 WC ) Seeds 32 Events Singles men women boys girls Doubles men women mixed boys girls Legends men women mixed WC Singles men women quad WC Doubles men women quad ← 2017 Australian Open 2019 →
  Fact 1.0: The men's singles champion of the 2018 Australian Open was Roger Federer.
  Fact 1.1: The men's singles runner-up of the 2018 Australian Open was Marin Čilić.
  Fact 1.2: The 2018 Australian Open men's singles final score was 6-2, 6-7, 6-3, 3-6, 6-1.
  Fact 1.3: The 2018 Australian Open men's singles draw had 128 entries (16 qualifiers and 8 wild cards).
  Fact 1.4: The 2018 Australian Open men's singles had 32 seeds.
  Fact 1.5: The previous edition was the 2017 Australian Open.
  Fact 1.6: The next edition was the 2019 Australian Open.

Retrieved items:
<item 1>
2018 Australian Open – Men's Singles
2018 Australian Open -- men 's singles - wikipedia 2018 Australian Open -- men 's singles Jump to : navigation , search Men 's Singles 2018 Australian Open
Champion Roger Federer Runner - up Marin Čilić Final score 6 -- 2 , 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 Details Draw 128 ( 16 Q / 8 WC ) Seeds 32 Events Singles
men women boys girls Doubles men women mixed boys girls Legends men women mixed WC Singles men women quad WC Doubles men women quad ← 2017 Australian Open 2019
→ Main article : 2018 Australian Open Roger Federer was the defending champion and successfully defended his title , defeating Marin Čilić in the final , 6 -- 2
, 6 -- 7 , 6 -- 3 , 3 -- 6 , 6 -- 1 . It was Federer 's 20th Grand Slam singles title and record - equalling sixth Australian Open men 's singles title ( tied
with Roy Emerson and Novak Djokovic ) .
</item 1>

<item 2>
The text lists Australian Open men’s singles champions from 1969 to 2005, despite its 2018 heading. Multiple champions include Ken Rosewall, John Newcombe,
Guillermo Vilas, Johan Kriek, Mats Wilander, Stefan Edberg, Ivan Lendl, Jim Courier, Pete Sampras, and Andre Agassi. The tournament was not held in 1986. The
list ends with Roger Federer’s 2004 title and omits later results.
</item 2>

<item 3>
2018 Australian Open – Men's Singles
With the win , Federer became the first male player to win at least six titles at two Grand Slam tournaments ( six at the Australian Open and eight at Wimbledon
) . Federer became the oldest man to win a Grand Slam singles title in the Open era since Ken Rosewall in 1972 . The time span between Federer 's first Grand
slam glory at Wimbledon and this latest , almost 15 years , is an Open era record in the men 's singles field . This was also the 10th time that Federer has
defended a Grand Slam title , with the previous time being at the 2008 US Open . Čilić became the first Croatian player to reach a singles final at the
Australian Open . Federer and Rafael Nadal were in contention for the ATP no . 1 singles ranking at the start of the tournament . Nadal retained the top ranking
by reaching the quarterfinals .
</item 3>

<item 4>
2018 Australian Open – Men's Singles
2 Section 6 4.4. 3 Section 7 4.4. 4 Section 8 5 References Seeds ( edit ) 01 . Rafael Nadal ( Quarterfinals , retired ) 02 . Roger Federer ( Champion ) 03 .
Grigor Dimitrov ( Quarterfinals ) 04 . Alexander Zverev ( Third round ) 05 . Dominic Thiem ( Fourth round ) 06 . Marin Čilić ( Final ) 07 . David Goffin (
Second round ) 08 . Jack Sock ( First round ) 09 . Stan Wawrinka ( Second round ) 10 . Pablo Carreño Busta ( Fourth round ) 11 . Kevin Anderson ( First round )
12 . Juan Martín del Potro ( Third round ) 13 . Sam Querrey ( Second round ) 14 . Novak Djokovic ( Fourth round ) 15 . Jo - Wilfried Tsonga ( Third round ) 16 .
John Isner ( First round ) 17 .
</item 4>

<item 5>
Roger Federer defeated Marin Čilić to win the 2018 Australian Open men’s singles title. Coverage also highlighted Hyeon Chung’s rise from Milan to Melbourne and
his victory over Novak Djokovic. The excerpt includes related news citations and links to Australian Open men’s singles draws from the Open Era.
</item 5>

<item 6>
Roger Federer won the 2018 Australian Open, securing his sixth title there and eighth at Wimbledon—the first man to win six or more titles at two Grand Slam
events. He became the Open Era’s oldest men’s singles champion since Ken Rosewall in 1972, nearly 15 years after his first Slam title, and defended a major
title for the 10th time. Marin Čilić became the first Croatian Australian Open singles finalist. Rafael Nadal retained the world No.
</item 6>

<item 7>
Roger Federer successfully defended his 2018 Australian Open men’s singles title, beating Marin Čilić in a five-set final, 6–2, 6–7, 6–3, 3–6, 6–1. The victory
earned Federer his 20th Grand Slam singles title and a record-equalling sixth Australian Open men’s title, matching Roy Emerson and Novak Djokovic. The draw
featured 128 players, including 16 qualifiers and
</item 7>

<item 8>
2018 Australian Open – Men's Singles
1965 1966 1967 1968 Open era 1969 1970 1971 1972 1973 1976 1977 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998
1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Australian Open men 's singles champions ( 1969 ) Rod Laver ( 1970 )
Arthur Ashe ( 1971 ) Ken Rosewall ( 1972 ) Ken Rosewall ( 1973 ) John Newcombe ( 1974 ) Jimmy Connors ( 1975 ) John Newcombe ( 1976 ) Mark Edmondson ( 1977 (
Jan ) ) Roscoe Tanner ( 1977 ( Dec ) ) Vitas Gerulaitis (
</item 8>

<item 9>
2018 Australian Open – Men's Singles
Nadal retained the top ranking by reaching the quarterfinals . This was the first time since the 2008 Wimbledon Championships that two unseeded players ( Chung
Hyeon and Kyle Edmund ) reached the semifinals of the men 's singles event at a Grand Slam tournament , and the first time at the Australian Open since 1999 .
Chung became the first South Korean player to reach the quarterfinals and semifinals of a Grand Slam event . Djokovic 's fourth - round loss to Chung was his
first straight - sets loss at the Australian Open since 2007 . Contents ( hide ) 1 Seeds 2 Qualifying 3 Wildcards 4 Draw 4.1 Key 4.2 Finals 4.3 Top half 4.3. 1
Section 1 4.3. 2 Section 2 4.3. 3 Section 3 4.3. 4 Section 4 4.4 Bottom half 4.4. 1 Section 5 4.4. 2 Section 6 4.4. 3 Section 7 4.4.
</item 9>

<item 10>
Roger Federer won the 2018 Australian Open men’s singles title, having also won in 2017. The list names champions from 2003–2018, including Andre Agassi, Marat
Safin, Novak Djokovic, Rafael Nadal, and Stan Wawrinka. It also outlines 2018 ATP Tour events across Grand Slams, Masters 1000, 500, and 250 series, with
singles and doubles competitions.
</item 10>
=== END ITEM id=d6363c98187f2249 ===

=== ITEM id=938e7269dace9560 ===
Gold sentences and facts:
Sentence 0: Polar moment of inertia is a quantity used to predict an object 's ability to resist torsion , in objects ( or segments of objects ) with an invariant circular cross section and no significant warping or out - of - plane deformation . It is used to calculate the angular displacement of an object subjected to a torque . It is analogous to the area moment of inertia , which characterizes an object 's ability to resist bending and is required to calculate displacement .
  Fact 0.0: Polar moment of inertia is a quantity used to predict an object's ability to resist torsion.
  Fact 0.1: Polar moment of inertia predicts torsion resistance for objects, or segments of objects, with an invariant circular cross section.
  Fact 0.2: Polar moment of inertia predicts torsion resistance for objects with no significant warping or out-of-plane deformation.
  Fact 0.3: Polar moment of inertia is used to calculate the angular displacement of an object subjected to a torque.
  Fact 0.4: Polar moment of inertia is analogous to the area moment of inertia.
  Fact 0.5: The area moment of inertia characterizes an object's ability to resist bending.
  Fact 0.6: The area moment of inertia is required to calculate displacement.

Retrieved items:
<item 1>
Polar moment of inertia
See moment ( physics ) . Note : It has become common to use moment of inertia ( MOI ) to refer to either or both of the planar second moment of area , I = ∫ A x
2 d A ( \ displaystyle I = \ int _ ( A ) x ^ ( 2 ) \ , \ mathrm ( d ) A ) , where x is the distance to some reference plane , or the polar second moment of area
, I = ∫ A r 2 d A ( \ displaystyle I = \ int _ ( A ) r ^ ( 2 ) \ , \ mathrm ( d ) A ) , where r is the distance to some reference axis . In each case the
integral is over all the infinitesimal elements of area , dA , in some two - dimensional cross-section .
</item 1>

<item 2>
Polar moment of inertia
Englewood Cliffs , NJ . 1995 . ISBN 0 - 13 - 137589 - X . Jump up ^ http://www.efunda.com/math/areas/MomentOfInertia.cfm Jump up ^ Blodgeit . Design of Welded
Structures ( 1982 ) . Section 2.10. 2 . Jump up ^ `` What is the difference between the Polar Moment of Inertia , IPIP and the torsional constant , JTJT of a
cross section ? '' .
</item 2>

<item 3>
Polar moment of inertia
I z = ∫ A ρ 2 d A ( \ displaystyle I_ ( z ) = \ int _ ( A ) \ rho ^ ( 2 ) \ , dA ) I = the polar moment of area about the axis z dA = an elemental area ρ = the
radial distance to the element dA from the axis z For a circular section with radius r : I z = ∫ 0 2 π ∫ 0 r ρ 2 ρ d ρ d φ = π r 4 2 ( \ displaystyle I_ ( z ) =
\ int _ ( 0 ) ^ ( 2 \ pi ) \ int _ ( 0 ) ^ ( r ) \ rho ^ ( 2 ) \ rho \ , d \ rho \ , d \ phi = ( \ frac ( \ pi r ^ ( 4 ) ) ( 2 ) ) ) Unit ( edit ) The SI unit
for polar moment of area , like the area moment of inertia , is meter to the fourth power ( m ) .
</item 3>

<item 4>
Polar moment of inertia
Conversion from area moment of inertia ( edit ) By the perpendicular axis theorem , the following equation relates I to the area moments of inertia about the
other two mutually perpendicular axes : I z = I x + I y ( \ displaystyle I_ ( z ) = I_ ( x ) + I_ ( y ) ) I x = I y ( \ displaystyle I_ ( x ) = I_ ( y ) ) is
only true for a circular or hollow circular section with a constant thickness . For non circular cross section analytical formulas are needed : torsion constant
. Second moment of inertia , I , is the resistance of a solid to being twisted . Polar moment of inertia , I or J , is the inertia moment of a rotating solid
Application ( edit ) The polar moment of area appears in the formulae that describe torsional stress and angular displacement .
</item 4>

<item 5>
Polar moment of inertia
Torsional stresses : τ = T r J z ( \ displaystyle \ tau = ( \ frac ( Tr ) ( J_ ( z ) ) ) ) where T ( \ displaystyle T ) is the torque , r ( \ displaystyle r )
is the distance from the center and J z ( \ displaystyle J_ ( z ) ) is the polar moment of area . In a circular shaft , the shear stress is maximal at the
surface of the shaft . Sample calculation ( edit ) The rotor of a modern steam turbine . Calculation of the steam turbine shaft radius for a turboset :
Assumptions : Power carried by the shaft is 1000 MW ; this is typical for a large nuclear power plant . Yield stress of the steel used to make the shaft ( τ )
is : 250 × 10 N / m2 . Electricity has a frequency of 50 Hz ; this is the typical frequency in Europe . In North America the frequency is 60 Hz . This is
assuming that there is a 1 : 1 correlation between rotational velocity of turbine and the frequency of mains power .
</item 5>

<item 6>
Polar moment of inertia
In such cases , the torsion constant can be substituted instead . In objects with significant cross-sectional variation ( along the axis of the applied torque )
, which can not be analyzed in segments , a more complex approach may have to be used . See 3 - D elasticity . However the polar moment of area can be used to
calculate the moment of inertia of an object with arbitrary cross-section . Definition ( edit ) A schematic showing how the polar moment of area is calculated
for an arbitrary shape about an axis o. ρ is the radial distance to the element dA .
</item 6>

<item 7>
Polar moment of inertia
Moment of inertia is , strictly , the second moment of mass with respect to distance from an axis : I = ∫ m r 2 d m ( \ displaystyle I = \ int _ ( m ) r ^ ( 2 )
\ mathrm ( d ) m ) , where r is the distance to some potential rotation axis , and the integral is over all the infinitesimal elements of mass , dm , in a three
- dimensional space occupied by an object . The MOI , in this sense , is the analog of mass for rotational problems . Contents ( hide ) 1 Limitations 2
Definition 3 Unit 4 Conversion from Area Moment of Inertia 5 Application 6 Sample calculation 7 Comparing various moments of inertias for a cylinder 7.1 Polar
moment of inertia 8 See also 9 References 10 External links Limitations ( edit ) The polar moment of area can not be used to analyze shafts with non-circular
cross-sections . In such cases , the torsion constant can be substituted instead .
</item 7>

<item 8>
Polar moment of inertia
Polar moment of inertia - wikipedia Polar moment of inertia Jump to : navigation , search Polar moment of inertia is a quantity used to predict an object 's
ability to resist torsion , in objects ( or segments of objects ) with an invariant circular cross section and no significant warping or out - of - plane
deformation . It is used to calculate the angular displacement of an object subjected to a torque . It is analogous to the area moment of inertia , which
characterizes an object 's ability to resist bending and is required to calculate displacement . The larger the polar moment of area , the less the beam will
twist , when subjected to a given torque . Polar moment of area should not be confused with moment of inertia , which characterizes an object 's angular
acceleration due to a torque . See moment ( physics ) .
</item 8>

<item 9>
Polar moment of inertia
. first1 = missing last1 = in Authors list ( help ) External links ( edit ) Torsion of Shafts - engineeringtoolbox.com Elastic Properties and Young Modulus for
some Materials - engineeringtoolbox.com Retrieved from `` https://en.wikipedia.org/w/index.php?title=Polar_moment_of_inertia&oldid=805920942 '' Categories :
Continuum mechanics Mechanics Hidden categories : CS1 errors : missing author or editor Talk Contents About Wikipedia Eesti 한국어 עברית Magyar Русский Simple
English Українська 中文 Edit links This page was last edited on 18 October 2017 , at 13 : 57 . About Wikipedia
</item 9>

<item 10>
Polar moment of inertia
= π D 4 32 ( \ displaystyle I_ ( z ) = ( \ frac ( \ pi D ^ ( 4 ) ) ( 32 ) ) ) Mass moment of inertia J z = I z ρ l = π ρ l D 4 32 ( \ displaystyle J_ ( z ) = I_
( z ) \ rho l = ( \ frac ( \ pi \ rho lD ^ ( 4 ) ) ( 32 ) ) ) where : d ( \ displaystyle d ) is the inner diameter in m D ( \ displaystyle D ) is the outside
diameter in m J z ( \ displaystyle J_ ( z ) ) is the mass moment of inertia in kg m l ( \ displaystyle l ) is the length of cylinder in m ρ ( \ displaystyle \
rho ) is the specific mass in kg / m See also ( edit ) Torsion constant Torsion spring Torsion bar Area moment of inertia References ( edit ) Jump up ^ Ugural
AC , Fenster SK . Advanced Strength and Applied Elasticity . 3rd Ed . Prentice - Hall Inc . Englewood Cliffs , NJ . 1995 .
</item 10>
=== END ITEM id=938e7269dace9560 ===

=== ITEM id=29431eac7ddbe348 ===
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
Modern biology and biochemistry make intensive use of these techniques in recombinant DNA technology . Recombinant DNA is a man - made DNA sequence that has
been assembled from other DNA sequences . They can be transformed into organisms in the form of plasmids or in the appropriate format , by using a viral vector
. The genetically modified organisms produced can be used to produce products such as recombinant proteins , used in medical research , or be grown in
agriculture . DNA profiling Further information : DNA profiling Forensic scientists can use DNA in blood , semen , skin , saliva or hair found at a crime scene
to identify a matching DNA of an individual , such as a perpetrator . This process is formally termed DNA profiling , but may also be called `` genetic
fingerprinting '' . In DNA profiling , the lengths of variable sections of repetitive DNA , such as short tandem repeats and minisatellites , are compared
between people . This method is usually an extremely reliable technique for identifying a matching DNA . However , identification can be complicated if the
scene is contaminated with DNA from several people .
</item 1>

<item 2>
DNA
DNA - wikipedia DNA Jump to : navigation , search For a non-technical introduction to the topic , see Introduction to genetics . For other uses , see DNA (
disambiguation ) . The structure of the DNA double helix . The atoms in the structure are colour - coded by element and the detailed structures of two base
pairs are shown in the bottom right . The structure of part of a DNA double helix Deoxyribonucleic acid ( / diˈɒksiˌraɪboʊnjʊˈkliːɪk , - ˈkleɪ. ɪk / ( listen )
; DNA ) is a thread - like chain of nucleotides carrying the genetic instructions used in the growth , development , functioning and reproduction of all known
living organisms and many viruses . DNA and ribonucleic acid ( RNA ) are nucleic acids ; alongside proteins , lipids and complex carbohydrates ( polysaccharides
) , they are one of the four major types of macromolecules that are essential for all known forms of life . Most DNA molecules consist of two biopolymer strands
coiled around each other to form a double helix .
</item 2>

<item 3>
DNA
`` Jump up ^ Miescher F ( 1871 ) . `` Ueber die chemische Zusammensetzung der Eiterzellen '' ( On the chemical composition of pus cells ) . Medicinisch -
chemische Untersuchungen ( in German ) . 4 : 441 -- 460 . Ich habe mich daher später mit meinen Versuchen an die ganzen Kerne gehalten , die Trennung der Körper
, die ich einstweilen ohne weiteres Präjudiz als lösliches und unlösliches Nuclein bezeichnen will , einem günstigeren Material überlassend . ( Therefore , in
my experiments I subsequently limited myself to the whole nucleus , leaving to a more favorable material the separation of the substances , that for the present
, without further prejudice , I will designate as soluble and insoluble nuclear material ( `` Nuclein '' ) Jump up ^ Dahm R ( January 2008 ) . `` Discovering
DNA : Friedrich Miescher and the early years of nucleic acid research '' . Human Genetics . 122 ( 6 ) : 565 -- 81 .
</item 3>

<item 4>
DNA
Genes and genomes Further information : Cell nucleus , Chromatin , Chromosome , Gene , and Noncoding DNA Genomic DNA is tightly and orderly packed in the
process called DNA condensation , to fit the small available volumes of the cell . In eukaryotes , DNA is located in the cell nucleus , with small amounts in
mitochondria and chloroplasts . In prokaryotes , the DNA is held within an irregularly shaped body in the cytoplasm called the nucleoid . The genetic
information in a genome is held within genes , and the complete set of this information in an organism is called its genotype . A gene is a unit of heredity and
is a region of DNA that influences a particular characteristic in an organism . Genes contain an open reading frame that can be transcribed , and regulatory
sequences such as promoters and enhancers , which control transcription of the open reading frame . In many species , only a small fraction of the total
sequence of the genome encodes protein . For example , only about 1.5 % of the human genome consists of protein - coding exons , with over 50 % of human DNA
consisting of non-coding repetitive sequences .
</item 4>

<item 5>
DNA
In this way , the base on the old strand dictates which base appears on the new strand , and the cell ends up with a perfect copy of its DNA . Extracellular
nucleic acids Naked extracellular DNA ( eDNA ) , most of it released by cell death , is nearly ubiquitous in the environment . Its concentration in soil may be
as high as 2 μg / L , and its concentration in natural aquatic environments may be as high at 88 μg / L. Various possible functions have been proposed for eDNA
: it may be involved in horizontal gene transfer ; it may provide nutrients ; and it may act as a buffer to recruit or titrate ions or antibiotics .
Extracellular DNA acts as a functional extracellular matrix component in the biofilms of several bacterial species . It may act as a recognition factor to
regulate the attachment and dispersal of specific cell types in the biofilm ; it may contribute to biofilm formation ; and it may contribute to the biofilm 's
physical strength and resistance to biological stress . Cell - free fetal DNA is found in the blood of the mother , and can be sequenced to determine a great
deal of information about the developing fetus .
</item 5>

<item 6>
DNA
This information is replicated as and when the two strands separate . A large part of DNA ( more than 98 % for humans ) is non-coding , meaning that these
sections do not serve as patterns for protein sequences . The two strands of DNA run in opposite directions to each other and are thus antiparallel . Attached
to each sugar is one of four types of nucleobases ( informally , bases ) . It is the sequence of these four nucleobases along the backbone that encodes
biological information . RNA strands are created using DNA strands as a template in a process called transcription . Under the genetic code , these RNA strands
are translated to specify the sequence of amino acids within proteins in a process called translation . Within eukaryotic cells , DNA is organized into long
structures called chromosomes . During cell division these chromosomes are duplicated in the process of DNA replication , providing each cell its own complete
set of chromosomes . Eukaryotic organisms ( animals , plants , fungi and protists ) store most of their DNA inside the cell nucleus and some of their DNA in
organelles , such as mitochondria or chloroplasts .
</item 6>

<item 7>
DNA
Biological functions Location of eukaryote nuclear DNA within the chromosomes . DNA usually occurs as linear chromosomes in eukaryotes , and circular
chromosomes in prokaryotes . The set of chromosomes in a cell makes up its genome ; the human genome has approximately 3 billion base pairs of DNA arranged into
46 chromosomes . The information carried by DNA is held in the sequence of pieces of DNA called genes . Transmission of genetic information in genes is achieved
via complementary base pairing . For example , in transcription , when a cell uses the information in a gene , the DNA sequence is copied into a complementary
RNA sequence through the attraction between the DNA and the correct RNA nucleotides . Usually , this RNA copy is then used to make a matching protein sequence
in a process called translation , which depends on the same interaction between RNA nucleotides . In alternative fashion , a cell may simply copy its genetic
information in a process called DNA replication . The details of these functions are covered in other articles ; here the focus is on the interactions between
DNA and other molecules that mediate the function of the genome .
</item 7>

<item 8>
DNA
In contrast prokaryotes ( bacteria and archaea ) store their DNA only in the cytoplasm . Within the eukaryotic chromosomes , chromatin proteins such as histones
compact and organize DNA . These compact structures guide the interactions between DNA and other proteins , helping control which parts of the DNA are
transcribed . DNA was first isolated by Friedrich Miescher in 1869 . Its molecular structure was first identified by James Watson and Francis Crick at the
Cavendish Laboratory within the University of Cambridge in 1953 , whose model - building efforts were guided by X-ray diffraction data acquired by Raymond
Gosling , who was a post-graduate student of Rosalind Franklin . DNA is used by researchers as a molecular tool to explore physical laws and theories , such as
the ergodic theorem and the theory of elasticity . The unique material properties of DNA have made it an attractive molecule for material scientists and
engineers interested in micro - and nano - fabrication . Among notable advances in this field are DNA origami and DNA - based hybrid materials .
</item 8>

<item 9>
DNA
This can be used in studies ranging from ecological genetics to anthropology ; For example , DNA evidence is being used to try to identify the Ten Lost Tribes
of Israel . Information storage Main article : DNA digital data storage In a paper published in Nature in January 2013 , scientists from the European
Bioinformatics Institute and Agilent Technologies proposed a mechanism to use DNA 's ability to code information as a means of digital data storage . The group
was able to encode 739 kilobytes of data into DNA code , synthesize the actual DNA , then sequence the DNA and decode the information back to its original form
, with a reported 100 % accuracy . The encoded information consisted of text files and audio files . A prior experiment was published in August 2012 . It was
conducted by researchers at Harvard University , where the text of a 54,000 - word book was encoded in DNA . Moreover , in living cells , the storage can be
turned active by enzymes . Light - gated protein domains fused to DNA processing enzymes are suitable for that task in vitro . Fluorescent exonucleases can
transmit the output according to the nucleotide they have read .
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=29431eac7ddbe348 ===

=== ITEM id=64f8e0223f886e4d ===
Gold sentences and facts:
Sentence 0: Coordinates : 90 ° N 0 ° W ﻿ / ﻿ 90 ° N - 0 ° E ﻿ / 90 ; - 0
  Fact 0.0: The coordinates are listed as 90 degrees N, 0 degrees W.
  Fact 0.1: The coordinates are listed as 90 degrees N, 0 degrees E.
  Fact 0.2: The coordinates are listed as 90, -0.
Sentence 1: The North Pole is the northernmost point on the Earth , lying diametrically opposite the South Pole . It defines geodetic latitude 90 ° North , as well as the direction of true north . At the North Pole all directions point south ; all lines of longitude converge there , so its longitude can be defined as any degree value . Along tight latitude circles , counterclockwise is east and clockwise is west . The North Pole is at the center of the Northern Hemisphere .
  Fact 1.0: The North Pole is the northernmost point on the Earth.
  Fact 1.1: The North Pole lies diametrically opposite the South Pole.
  Fact 1.2: The North Pole defines geodetic latitude 90 degrees North.
  Fact 1.3: The North Pole defines the direction of true north.
  Fact 1.4: At the North Pole all directions point south.
  Fact 1.5: At the North Pole all lines of longitude converge.
  Fact 1.6: The longitude of the North Pole can be defined as any degree value.
  Fact 1.7: Along tight latitude circles, counterclockwise is east.
  Fact 1.8: Along tight latitude circles, clockwise is west.
  Fact 1.9: The North Pole is at the center of the Northern Hemisphere.

Retrieved items:
<item 1>
Graph query row:
entity: North Pole
relationship: LOCATED_IN
relation: None
fact: None
context: None
related: Arctic Ocean
</item 1>

<item 2>
North Pole
^ Data is from a Greenlandic Weather Station at 83 ° 38 ′ N 033 ° 22 ′ W ﻿ / ﻿ 83.633 ° N 33.367 ° W ﻿ / 83.633 ; - 33.367 ﻿ ( Greenlandic Weather Station )
located 709 km ( 441 mi ) from the North Pole . References ( edit ) Jump up ^ Black , Richard ( 8 April 2001 ) . New warning on Arctic sea ice melt . BBC Jump
up ^ Ljunggren , David ( 5 March 2009 ) . Arctic summer ice could vanish by 2013 : expert . Reuters Jump up ^ Russian sub plants flag at North Pole , Reuters ,
2 August 2007 Jump up ^ Андерсон , Уильям Роберт ( 1965 ) . `` '' Наутилус '' у Северного полюса `` . Воениздат . Retrieved 12 January 2012 . Jump up ^ Mouton ,
M.W. ( 1968 ) . The International Regime of the Polar Regions .
</item 2>

<item 3>
South Pole
The geographic coordinates of the South Pole are usually given simply as 90 ° S , since its longitude is geometrically undefined and irrelevant . When a
longitude is desired , it may be given as 0 ° . At the South Pole , all directions face north . For this reason , directions at the Pole are given relative to
`` grid north '' , which points northwards along the prime meridian . Along tight latitude circles , clockwise is east , and counterclockwise is west , opposite
to the North Pole . The Geographic South Pole is located on the continent of Antarctica ( although this has not been the case for all of Earth 's history
because of continental drift ) . It sits atop a featureless , barren , windswept and icy plateau at an altitude of 2,835 metres ( 9,301 ft ) above sea level ,
and is located about 1,300 km ( 800 mi ) from the nearest open sea at Bay of Whales . The ice is estimated to be about 2,700 metres ( 9,000 ft ) thick at the
Pole , so the land surface under the ice sheet is actually near sea level .
</item 3>

<item 4>
North Pole
2 MLAE 2009 Expedition 2.4. 3 MLAE 2013 Expedition 3 Day and night 4 Time 5 Climate 6 Flora and fauna 7 Territorial claims to the North Pole and Arctic regions
8 Cultural associations 9 See also 10 Notes 11 References 12 Further reading 13 External links Precise definition ( edit ) See also : Polar motion The Earth 's
axis of rotation -- and hence the position of the North Pole -- was commonly believed to be fixed ( relative to the surface of the Earth ) until , in the 18th
century , the mathematician Leonhard Euler predicted that the axis might `` wobble '' slightly . Around the beginning of the 20th century astronomers noticed a
small apparent `` variation of latitude , '' as determined for a fixed point on Earth from the observation of stars . Part of this variation could be attributed
to a wandering of the Pole across the Earth 's surface , by a range of a few metres . The wandering has several periodic components and an irregular component .
</item 4>

<item 5>
North Pole
There is no permanent human presence at the North Pole and no particular time zone has been assigned . Polar expeditions may use any time zone that is
convenient , such as Greenwich Mean Time , or the time zone of the country from which they departed . Climate ( edit ) Arctic ice shrinkages of 2007 compared to
2005 and also compared to the 1979 -- 2000 average . Main article : Climate of the Arctic See also : Climate change in the Arctic The North Pole is
substantially warmer than the South Pole because it lies at sea level in the middle of an ocean ( which acts as a reservoir of heat ) , rather than at altitude
on a continental land mass . Despite being an ice cap , it shares some characteristics with a tundra climate ( ETf ) due to the July and August temperatures
peaking just above freezing . Winter temperatures at the North Pole can range from about − 50 to − 13 ° C ( − 58 to 9 ° F ) , averaging around − 31 ° C ( − 24 °
F ) .
</item 5>

<item 6>
North Pole
Finding the North Pole . The Lyons press . ISBN 1 - 59228 - 072 - 2 . CS1 maint : Multiple names : authors list ( link ) Pal , Christopher ( 2002 ) . The Oddest
Place on Earth : Rediscovering the North Pole . Writer 's Showcase . ISBN 0 - 595 - 21454 - 1 . Fairfield , William ( 1885 ) . Paradise Found a Cradle of the
Human Race at the North Pole a Study of the Prehistoric World . Houghton , Mifflin . ISBN 0 - 665 - 37759 - 2 . External links ( edit ) Wikimedia Commons has
media related to North Pole . Arctic Council The Northern Forum North Pole Web Cam FAQ on the Arctic and the North Pole Daylight , Darkness and Changing of the
Seasons at the North Pole Video of the Nuclear Icebreaker Yamal visiting the North Pole in 2001 Polar Discovery : North Pole Observatory Expedition Polar
exploration Arctic Ocean History Expeditions Research stations Farthest North North Pole Barentsz Hudson Marmaduke Carolus Parry North Magnetic Pole J. Ross
J.C.
</item 6>

<item 7>
North Pole
North Pole - wikipedia North Pole Jump to : navigation , search For other uses , see North Pole ( disambiguation ) . Not to be confused with North Magnetic Pole
. Coordinates : 90 ° N 0 ° W ﻿ / ﻿ 90 ° N - 0 ° E ﻿ / 90 ; - 0 An azimuthal projection showing the Arctic Ocean and the North Pole . The map also shows the 75th
parallel north and 60th parallel north . Sea ice at the North Pole in 2006 The North Pole , also known as the Geographic North Pole or Terrestrial North Pole ,
is ( subject to the caveats explained below ) defined as the point in the Northern Hemisphere where the Earth 's axis of rotation meets its surface . The North
Pole is the northernmost point on the Earth , lying diametrically opposite the South Pole . It defines geodetic latitude 90 ° North , as well as the direction
of true north . At the North Pole all directions point south ; all lines of longitude converge there , so its longitude can be defined as any degree value .
Along tight latitude circles , counterclockwise is east and clockwise is west .
</item 7>

<item 8>
North Pole
Along tight latitude circles , counterclockwise is east and clockwise is west . The North Pole is at the center of the Northern Hemisphere . While the South
Pole lies on a continental land mass , the North Pole is located in the middle of the Arctic Ocean amid waters that are almost permanently covered with
constantly shifting sea ice . This makes it impractical to construct a permanent station at the North Pole ( unlike the South Pole ) . However , the Soviet
Union , and later Russia , constructed a number of manned drifting stations on a generally annual basis since 1937 , some of which have passed over or very
close to the Pole . Since 2002 , the Russians have also annually established a base , Barneo , close to the Pole . This operates for a few weeks during early
spring . Studies in the 2000s predicted that the North Pole may become seasonally ice - free because of Arctic ice shrinkage , with timescales varying from 2016
to the late 21st century or later .
</item 8>

<item 9>
North Pole
Exploration ( edit ) See also : Arctic exploration , Farthest North , List of Arctic expeditions , and List of firsts in the Geographic North Pole Pre-1900 (
edit ) Gerardus Mercator 's map of the North Pole from 1595 C.G. Zorgdragers map of the North Pole from 1720 As early as the 16th century , many prominent
people correctly believed that the North Pole was in a sea , which in the 19th century was called the Polynya or Open Polar Sea . It was therefore hoped that
passage could be found through ice floes at favorable times of the year . Several expeditions set out to find the way , generally with whaling ships , already
commonly used in the cold northern latitudes . One of the earliest expeditions to set out with the explicit intention of reaching the North Pole was that of
British naval officer William Edward Parry , who in 1827 reached latitude 82 ° 45 ′ North . In 1871 the Polaris expedition , a US attempt on the Pole led by
Charles Francis Hall , ended in disaster .
</item 9>

<item 10>
North Pole
The sea depth at the North Pole has been measured at 4,261 m ( 13,980 ft ) by the Russian Mir submersible in 2007 and at 4,087 m ( 13,410 ft ) by USS Nautilus
in 1958 . The nearest land is usually said to be Kaffeklubben Island , off the northern coast of Greenland about 700 km ( 430 mi ) away , though some perhaps
semi permanent gravel banks lie slightly closer . The nearest permanently inhabited place is Alert in the Qikiqtaaluk Region , Nunavut , Canada , which is
located 817 km ( 508 mi ) from the Pole . Contents ( hide ) 1 Precise definition 2 Exploration 2.1 Pre-1900 2.2 1900 -- 40 2.3 1940 -- 2000 2.4 21st century
2.4. 1 2007 descent to the North Pole seabed 2.4. 2 MLAE 2009 Expedition 2.4.
</item 10>
=== END ITEM id=64f8e0223f886e4d ===

=== ITEM id=261105e46e7be99a ===
Gold sentences and facts:
Sentence 0: The gymnosperms are a group of seed - producing plants that includes conifers , cycads , Ginkgo , and gnetophytes . The term `` gymnosperm '' comes from the Greek composite word γυμνόσπερμος ( γυμνός gymnos , `` naked '' and σπέρμα sperma , `` seed '' ) , meaning `` naked seeds '' . The name is based on the unenclosed condition of their seeds ( called ovules in their unfertilized state ) . The non-encased condition of their seeds stands in contrast to the seeds and ovules of flowering plants ( angiosperms ) , which are enclosed within an ovary . Gymnosperm seeds develop either on the surface of scales or leaves , which are often modified to form cones , or solitary as in Yew , Torreya , Ginkgo .
  Fact 0.0: The gymnosperms are a group of seed-producing plants.
  Fact 0.1: Gymnosperms include conifers, cycads, Ginkgo, and gnetophytes.
  Fact 0.2: The term gymnosperm comes from the Greek composite word γυμνόσπερμος.
  Fact 0.3: The Greek word γυμνός (gymnos) means naked.
  Fact 0.4: The Greek word σπέρμα (sperma) means seed.
  Fact 0.5: The term gymnosperm means naked seeds.
  Fact 0.6: The name gymnosperm is based on the unenclosed condition of gymnosperm seeds.
  Fact 0.7: Gymnosperm ovules are called ovules in their unfertilized state.
  Fact 0.8: Gymnosperm seeds are not encased.
  Fact 0.9: The seeds and ovules of flowering plants (angiosperms) are enclosed within an ovary.
  Fact 0.10: Gymnosperm seeds develop either on the surface of scales or leaves.
  Fact 0.11: The scales or leaves on which gymnosperm seeds develop are often modified to form cones.
  Fact 0.12: Gymnosperm seeds can also be solitary, as in Yew, Torreya, and Ginkgo.

Retrieved items:
<item 1>
Gymnosperm
The name is based on the unenclosed condition of their seeds ( called ovules in their unfertilized state ) . The non-encased condition of their seeds stands in
contrast to the seeds and ovules of flowering plants ( angiosperms ) , which are enclosed within an ovary . Gymnosperm seeds develop either on the surface of
scales or leaves , which are often modified to form cones , or solitary as in Yew , Torreya , Ginkgo . The gymnosperms and angiosperms together compose the
spermatophytes or seed plants . The gymnosperms are divided into six phyla . Organisms that belong to the Cycadophyta , Ginkgophyta , Gnetophyta , and Pinophyta
( also known as Coniferophyta ) phyla are still in existence while those in the Pteridospermales and Cordaitales phyla are now extinct .
</item 1>

<item 2>
Gymnosperm
Diselma , Fitzroya , Callitris ( incl . Actinostrobus and Neocallitropsis ) , Thujopsis , Thuja , Fokienia , Chamaecyparis , Callitropsis , Cupressus ,
Juniperus , Xanthocyparis , Calocedrus , Tetraclinis , Platycladus , Microbiota Family Taxaceae : Austrotaxus , Pseudotaxus , Taxus , Cephalotaxus , Amentotaxus
, Torreya Diversity and origin ( edit ) There are more than 1000 extant or currently living species of gymnosperms in 88 plant genera belonging to 14 plant
families . It is widely accepted that the gymnosperms originated in the late Carboniferous period , replacing the lycopsid rainforests of the tropical region .
This appears to have been the result of a whole genome duplication event around 319 million years ago . Early characteristics of seed plants were evident in
fossil progymnosperms of the late Devonian period around 383 million years ago .
</item 2>

<item 3>
Gymnosperm
Agathis in Araucariaceae and Nageia in Podocarpaceae have broad , flat strap - shaped leaves . Cycads are the next most abundant group of gymnosperms , with two
or three families , 11 genera , and approximately 338 species . A majority of cycads are native to tropical climates and are most abundantly found in regions
near the equator . The other extant groups are the 95 - 100 species of Gnetales and one species of Ginkgo . Uses ( edit ) Gymnosperms have major economic uses .
Pine , fir , spruce , and cedar are all examples of conifers that are used for lumber , paper production , and resin . Some other common uses for gymnosperms
are soap , varnish , nail polish , food , gum , and perfumes . Life cycle ( edit ) Example of gymnosperm lifecycle Gymnosperms , like all vascular plants , have
a sporophyte - dominant life cycle , which means they spend most of their life cycle with diploid cells , while the gametophyte ( gamete - bearing phase ) is
relatively short - lived .
</item 3>

<item 4>
Gymnosperm
After syngamy ( joining of the sperm and egg cell ) , the zygote develops into an embryo ( young sporophyte ) . More than one embryo is usually initiated in
each gymnosperm seed . The mature seed comprises the embryo and the remains of the female gametophyte , which serves as a food supply , and the seed coat .
Genetics ( edit ) The first published sequenced genome for any gymnosperm was the genome of Picea abies in 2013 . References ( edit ) ^ Jump up to : ``
Gymnosperms on The Plant List '' . Theplantlist.org . Retrieved 2013 - 07 - 24 . ^ Jump up to : Raven , P.H. ( 2013 ) . Biology of Plants . New York : W.H.
Freeman and Co . Jump up ^ Jeffrey D. Palmer ; Douglas E. Soltis ; Mark W. Chase ( 2004 ) . `` The plant tree of life : an overview and some points of view '' .
American Journal of Botany. 91 ( 10 ) : 1437 -- 1445 .
</item 4>

<item 5>
Gymnosperm
Two spore types , microspores and megaspores , are typically produced in pollen cones or ovulate cones , respectively . Gametophytes , as with all heterosporous
plants , develop within the spore wall . Pollen grains ( microgametophytes ) mature from microspores , and ultimately produce sperm cells . Megagametophytes
develop from megaspores and are retained within the ovule . Gymnosperms produce multiple archegonia , which produce the female gamete . During pollination ,
pollen grains are physically transferred between plants from the pollen cone to the ovule . Pollen is usually moved by wind or insects . Whole grains enter each
ovule through a microscopic gap in the ovule coat ( integument ) called the micropyle . The pollen grains mature further inside the ovule and produce sperm
cells . Two main modes of fertilization are found in gymnosperms . Cycads and Ginkgo have motile sperm that swim directly to the egg inside the ovule , whereas
conifers and gnetophytes have sperm with no flagella that are moved along a pollen tube to the egg .
</item 5>

<item 6>
Gymnosperm
91 ( 10 ) : 1437 -- 1445 . doi : 10.3732 / ajb. 91.10. 1437 . PMID 21652302 . Jump up ^ Stevens , P.F. `` Angiosperm Phylogeny Website - Seed Plant Evolution ''
. Jump up ^ Hilton , Jason , and Richard M. Bateman. 2006 . Pteridosperms are the backbone of seed - plant phylogeny . Journal of the Torrey Botanical Society
133 : 119 - 168 ( abstract ) Jump up ^ Christenhusz , M.J.M. ; Reveal , J.L. ; Farjon , A. ; Gardner , M.F. ; Mill , R.R. ; Chase , M.W. ( 2011 ) . `` A new
classification and linear sequence of extant gymnosperms '' ( PDF ) . Phytotaxa. 19 : 55 -- 70 . Jump up ^ Christenhusz , M.J.M. ; Byng , J.W. ( 2016 ) .
</item 6>

<item 7>
Gymnosperm
External links ( edit ) Wikimedia Commons has media related to Gymnosperm . Gymnosperm Database Gymnosperms on the Tree of Life Albert Seward ( 1911 ) . ``
Gymnosperms '' . Encyclopædia Britannica ( 11th ed . ) . Botany History of botany Subdisciplines Plant systematics Ethnobotany Paleobotany Plant anatomy Plant
ecology Phytogeography Geobotany Flora Phytochemistry Plant pathology Bryology Phycology Floristics Dendrology Plant groups Algae Archaeplastida Bryophyte
Non-vascular plants Vascular plants Spermatophytes Pteridophyte Gymnosperm Angiosperm Plant morphology ( glossary ) Plant cells Cell wall Phragmoplast Plastid
Plasmodesma Vacuole Tissues Meristem Vascular tissue Vascular bundle Ground tissue Mesophyll Cork Wood Storage organs Vegetative Root Rhizoid Bulb Rhizome Shoot
Stem Leaf Petiole Cataphyll Bud Sessility Reproductive ( Flower ) Flower development Inflorescence Umbel Raceme Bract Pedicellate Flower Whorl Floral symmetry
Floral diagram Floral formula Receptacle
</item 7>

<item 8>
Gymnosperm
There is conflicting evidence on the question of whether the living gymnosperms form a clade . The fossil record of gymnosperms includes many distinctive taxa
that do not belong to the four modern groups , including seed - bearing trees that have a somewhat fern - like vegetative morphology ( the so - called `` seed
ferns '' or pteridosperms . ) When fossil gymnosperms such as Bennettitales , Caytonia and the glossopterids are considered , it is clear that angiosperms are
nested within a larger gymnosperm clade , although which group of gymnosperms is their closest relative remains unclear . For the most recent classification on
extant gymnosperms see Christenhusz et al. ( 2011 ) . There are 12 families , 83 known genera with a total of ca 1080 known species ( Christenhusz & Byng 2016 )
.
</item 8>

<item 9>
Gymnosperm
By far the largest group of living gymnosperms are the conifers ( pines , cypresses , and relatives ) , followed by cycads , gnetophytes ( Gnetum , Ephedra and
Welwitschia ) , and Ginkgo biloba ( a single living species ) . Roots in some genera have fungal association with roots in the form of mycorrhiza ( Pinus ) ,
while in some others ( Cycas ) small specialised roots called coralloid roots are associated with nitrogen - fixing cyanobacteria . Contents ( hide ) 1
Classification 2 Diversity and origin 3 Uses 4 Life cycle 5 Genetics 6 References 7 External links Classification ( edit ) Further information : Spermatophyte
In early classification schemes , the gymnosperms ( Gymnospermae ) were regarded as a `` natural '' group . There is conflicting evidence on the question of
whether the living gymnosperms form a clade .
</item 9>

<item 10>
Gymnosperm
Gymnosperm - wikipedia Gymnosperm Jump to : navigation , search Gymnosperms Temporal range : 370 -- 0 Ma PreЄ Є Pg Devonian - Present Various gymnosperms .
Scientific classification Kingdom : Plantae Subkingdom : Embryophyta ( unranked ) : Gymnospermae ( paraphyletic ) Divisions Pinophyta ( or Coniferophyta ) -
Conifers Ginkgophyta - Ginkgo Cycadophyta - Cycads Gnetophyta - Gnetum , Ephedra , Welwitschia Encephalartos sclavoi cone , about 30 cm long The gymnosperms are
a group of seed - producing plants that includes conifers , cycads , Ginkgo , and gnetophytes . The term `` gymnosperm '' comes from the Greek composite word
γυμνόσπερμος ( γυμνός gymnos , `` naked '' and σπέρμα sperma , `` seed '' ) , meaning `` naked seeds '' .
</item 10>
=== END ITEM id=261105e46e7be99a ===

=== ITEM id=6381a6964002729c ===
Gold sentences and facts:
Sentence 0: Lion 's mane jellyfish Scientific classification Kingdom : Animalia Phylum : Cnidaria Class : Scyphozoa Order : Semaeostomeae Family : Cyaneidae Genus : Cyanea Species : C. capillata Binomial name Cyanea capillata ( Linnaeus , 1758 )
  Fact 0.0: The lion's mane jellyfish has the kingdom Animalia.
  Fact 0.1: The lion's mane jellyfish has the phylum Cnidaria.
  Fact 0.2: The lion's mane jellyfish has the class Scyphozoa.
  Fact 0.3: The lion's mane jellyfish has the order Semaeostomeae.
  Fact 0.4: The lion's mane jellyfish has the family Cyaneidae.
  Fact 0.5: The lion's mane jellyfish has the genus Cyanea.
  Fact 0.6: The lion's mane jellyfish has the species C. capillata.
  Fact 0.7: The binomial name of the lion's mane jellyfish is Cyanea capillata.
  Fact 0.8: The binomial name Cyanea capillata was given by Linnaeus in 1758.
Sentence 1: The lion 's mane jellyfish ( Cyanea capillata ) , also known as the giant jellyfish or the hair jelly , is the largest known species of jellyfish . Its range is confined to cold , boreal waters of the Arctic , northern Atlantic , and northern Pacific Oceans . It is common in the English Channel , Irish Sea , North Sea and in western Scandinavian waters south to Kattegat and Øresund . It may also drift in to the south - western part of the Baltic Sea ( where it can not breed due to the low salinity ) . Similar jellyfish -- which may be the same species -- are known to inhabit seas near Australia and New Zealand . The largest recorded specimen found washed up on the shore of Massachusetts Bay in 1870 , had a bell with a diameter of 2.3 metres ( 7 ft 6 in ) and tentacles 37.0 m ( 121.4 ft ) long . Lion 's mane jellyfish have been observed below 42 ° N latitude for some time in the larger bays of the east coast of the United States .
  Fact 1.0: The lion's mane jellyfish (Cyanea capillata) is also known as the giant jellyfish or the hair jelly.
  Fact 1.1: The lion's mane jellyfish is the largest known species of jellyfish.
  Fact 1.2: The range of the lion's mane jellyfish is confined to cold, boreal waters of the Arctic, northern Atlantic, and northern Pacific Oceans.
  Fact 1.3: The lion's mane jellyfish is common in the English Channel, Irish Sea, and North Sea.
  Fact 1.4: The lion's mane jellyfish is common in western Scandinavian waters south to Kattegat and Oresund.
  Fact 1.5: The lion's mane jellyfish may drift into the south-western part of the Baltic Sea.
  Fact 1.6: The lion's mane jellyfish cannot breed in the Baltic Sea due to the low salinity.
  Fact 1.7: Similar jellyfish, which may be the same species, are known to inhabit seas near Australia and New Zealand.
  Fact 1.8: The largest recorded specimen washed up on the shore of Massachusetts Bay in 1870.
  Fact 1.9: The largest recorded specimen had a bell with a diameter of 2.3 metres (7 ft 6 in).
  Fact 1.10: The largest recorded specimen had tentacles 37.0 m (121.4 ft) long.
  Fact 1.11: The lion's mane jellyfish has been observed below 42 degrees N latitude for some time.
  Fact 1.12: The lion's mane jellyfish has been observed in the larger bays of the east coast of the United States.

Retrieved items:
<item 1>
Lion's mane jellyfish
The largest recorded specimen found washed up on the shore of Massachusetts Bay in 1870 , had a bell with a diameter of 2.3 metres ( 7 ft 6 in ) and tentacles
37.0 m ( 121.4 ft ) long . Lion 's mane jellyfish have been observed below 42 ° N latitude for some time in the larger bays of the east coast of the United
States . The lion 's mane jellyfish uses its stinging tentacles to capture , pull in and eat prey such as fish , sea creatures and smaller jellyfish . Contents
( hide ) 1 Taxonomy 2 Description 3 Ecology 4 Behavior and reproduction 5 Sting and human contact 6 In popular culture 7 Gallery 8 References 8.1 Other sources
9 External links Taxonomy ( edit ) Cyanea sp . The taxonomy of the Cyanea species is not fully agreed upon ; some zoologists have suggested that all species
within the genus should be treated as one .
</item 1>

<item 2>
Lion's mane jellyfish
Behavior and reproduction ( edit ) Main article : Jellyfish § Life cycle Lion 's mane jellyfish swimming , side view Lion 's mane jellyfish remain mostly very
near the surface , at no more than 20 m depth . Their slow pulsations drive them forwards weakly , so they depend on ocean currents to travel great distances .
The jellyfish are most often spotted during the late summer and autumn , when they have grown to a large size and the currents begin to sweep them to shore .
Like other jellyfish , lion 's manes are capable of both sexual reproduction in the medusa stage and asexual reproduction in the polyp stage . Lion 's mane
jellyfish have four different stages in their year - long lifespan : a larval stage , a polyp stage , an ephyrae stage , and the medusa stage . The female
jellyfish carries its fertilized eggs in its tentacle , where the eggs grow into larva . When the larva are old enough , the female deposits them on a hard
surface , where the larva soon grow into polyps . The polyps begin to reproduce asexually , creating stacks of small creatures called ephyrae .
</item 2>

<item 3>
Lion's mane jellyfish
Two distinct taxa , however , occur together in at least the eastern North Atlantic , with the blue jellyfish ( Cyanea lamarckii Péron & Lesueur , 1810 )
differing in color ( blue , not red ) and smaller size ( 10 -- 20 cm diameter , rarely 35 cm ) . Populations in the western Pacific around Japan are sometimes
distinguished as Cyanea nozakii , or as a subspecies , C. c. nozakii . In 2015 , Russian researchers announced a possible sister species , Cyanea tzetlinii
found in the White Sea , but this has not yet been recognized by other authoritative databases such as WoRMS or ITIS . Description ( edit ) Lion 's mane
jellyfish ( Cyanea capillata ) are named for their showy , trailing tentacles reminiscent of a lion 's mane . They can vary greatly in size : although capable
of attaining a bell diameter of over 2 metres ( 6 ft 7 in ) , those found in lower latitudes are much smaller than their far northern counterparts , with a bell
about 50 centimetres ( 20 in ) in diameter .
</item 3>

<item 4>
Lion's mane jellyfish
Lion 's mane jellyfish - wikipedia Lion 's mane jellyfish Jump to : navigation , search Lion 's mane jellyfish Scientific classification Kingdom : Animalia
Phylum : Cnidaria Class : Scyphozoa Order : Semaeostomeae Family : Cyaneidae Genus : Cyanea Species : C. capillata Binomial name Cyanea capillata ( Linnaeus ,
1758 ) The lion 's mane jellyfish ( Cyanea capillata ) , also known as the giant jellyfish or the hair jelly , is the largest known species of jellyfish . Its
range is confined to cold , boreal waters of the Arctic , northern Atlantic , and northern Pacific Oceans . It is common in the English Channel , Irish Sea ,
North Sea and in western Scandinavian waters south to Kattegat and Øresund . It may also drift in to the south - western part of the Baltic Sea ( where it can
not breed due to the low salinity ) . Similar jellyfish -- which may be the same species -- are known to inhabit seas near Australia and New Zealand .
</item 4>

<item 5>
Lion's mane jellyfish
Polar Biol . doi : 10.1007 / s00300 - 015 - 1707 - y . Jump up ^ `` Lion 's Mane Jellyfish '' . Scubatravel.co.uk . Retrieved 2010 - 09 - 03 . Jump up ^
https://web.archive.org/web/20110108024017/http://www.typesofjellyfish.net/lions_mane_jellyfish/lions_mane_jellyfish.html Jump up ^ Kozloff , Eugene ( 2003 ) .
Seashore Life of the Northern Pacific Coast . University of Washington Press . pp. 54 , 56 ( 1983 edition ) . ISBN 0295960841 . Jump up ^ Point 6 at ^ Jump up
to : `` Lions Mane Jellyfish '' . Types Of Jellyfish . Archived from the original on 2011 - 01 - 08 . Retrieved 2010 - 09 - 03 .
</item 5>

<item 6>
Lion's mane jellyfish
Size seems to dictate coloration : larger specimens are a vivid crimson to dark purple while smaller specimens grade to a lighter orange or tan , occasionally
colorless . The bell of the lion 's mane jellyfish is scalloped into eight lobes ( lappets ) , each lobe containing from 70 to 150 tentacles , arranged in four
fairly distinct rows . Along the bell margin is a balance organ at each of the eight indentations between the lobes -- the rhopalium -- which helps the
jellyfish orient itself . From the central mouth extend broad frilly oral arms with many stinging cells . Closer to its mouth , its total number of tentacles
are around 1200 . The long , thin tentacles which emanate from the bell 's subumbrella have been characterised as `` extremely sticky '' ; they also have
stinging cells . The tentacles of larger specimens may trail as long as 30 m ( 100 ft ) or more , with the tentacles of the longest known specimen measured at
37 m ( 120 ft ) in length .
</item 6>

<item 7>
Lion's mane jellyfish
This was later corrected -- in 1864 , a bootlace worm ( Lineus longissimus ) was found washed up on a Scottish shore that was 55 m ( 180 ft ) long . ( That
claim has its critics : because bootlace worms can easily stretch to several times their natural length , it is possible the worm did not actually grow to be
that length . ) Gallery ( edit ) A lion 's mane jelly capturing a sea gooseberry Lion 's mane jellyfish with full threads visible Lion 's mane jellyfish
contracted Lion 's mane jellyfish in star formation Lion 's mane jellyfish open , showing underside Life - sized model in the Smithsonian Museum of Natural
History , Washington DC , US Small , dead lion 's mane jelly washing up on the beach Blue lion 's mane jellyfish in Askeröfjorden outside Stenungsund , Sweden .
References ( edit ) Jump up ^ Hair Jelly ( Cyanea capillata ) . Australian Venom Research Unit . ^ Jump up to : `` Lion 's Mane Jellyfish -- Reference Library
'' . redOrbit . Retrieved 2010 - 09 - 03 .
</item 7>

<item 8>
Lion's mane jellyfish
redOrbit . Retrieved 2010 - 09 - 03 . Jump up ^ `` Lion 's Mane Jellyfish : Discovery Channel '' . Dsc.discovery.com. 2013 - 05 - 23 . Retrieved 2013 - 09 - 08
. Jump up ^ Glafira D. Kolbasova ; Arthur O. Zalevsky ; Azamat R. Gafurov ; Philipp O. Gusev ; Margarita A. Ezhova ; Anna A. Zheludkevich ; Olga P. Konovalova ;
Ksenia N. Kosobokova ; Nikita U. Kotlov ; Natalia O. Lanina ; Anna S. Lapashina ; Dmitry O. Medvedev ; Katerina S. Nosikova ; Ekaterina O. Nuzhdina ; Georgii A.
Bazykin ; Tatyana V. Neretina ( 2015 ) . `` A new species of Cyanea jellyfish sympatric to C. capillata in the White Sea '' . Polar Biol .
</item 8>

<item 9>
Lion's mane jellyfish
Considering the size of the species , it is possible that this mass incident was caused by a single specimen . In popular culture ( edit ) The lion 's mane
jellyfish appears in the Sherlock Holmes short story `` The Adventure of the Lion 's Mane '' published in The Case - Book of Sherlock Holmes . Holmes discovers
at the end of the story that the true killer of a professor who died shortly after going swimming ( shouting `` the lion 's mane '' before he succumbed ) was
actually this jellyfish . Suspicion was originally laid upon the professor 's rival in love , until the latter was similarly attacked ( he survived , although
badly stung ) . In the context of the story , it is only because the school professor has a weak heart that he succumbs , as is confirmed by the survival of the
second victim . A photograph widely distributed on the internet appears to show an anomalously large lion 's mane dwarfing a nearby diver by several times . The
photo was subsequently shown to be a hoax . On the popular television program QI the show claimed that the longest animal in the world was the lion 's mane
jellyfish .
</item 9>

<item 10>
Lion's mane jellyfish
wikipedia.org/w/index.php?title=Lion%27s_mane_jellyfish&oldid=834313563 '' Categories : Semaeostomeae Animals described in 1758 Hidden categories : Articles
with ' species ' microformats All articles with unsourced statements Articles with unsourced statements from February 2012 Talk Contents About Wikipedia
Wikispecies Azərbaycanca Català Cebuano Čeština Dansk Deutsch Español Esperanto فارسی Français Frysk Gaeilge 한국어 Bahasa Indonesia Íslenska Italiano עברית
Lietuvių Magyar Nederlands Norsk Norsk nynorsk پنجابی Polski Português Română Русский Simple English Suomi Svenska Türkçe Tiếng Việt Winaray 中文 26 more Edit
links This page was last edited on 5 April 2018 , at 01 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms
may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
</item 10>
=== END ITEM id=6381a6964002729c ===
