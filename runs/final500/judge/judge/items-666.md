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

=== ITEM id=0913458771e95c3b ===
Gold sentences and facts:
Sentence 0: A regulatory site ( also known as an `` allosteric site '' ) is a site on an allosteric protein to which a modulator molecule ( or allosteric regulator ) binds . This ligand - binding site on a receptor or enzyme is distinct from the active site . Allosteric modulators alter enzyme activity by binding to the regulatory site . Some allosteric regulators activate enzymes , while others inhibit them . The working hypothesis is that when allosteric regulators bind to the regulatory site , the protein undergoes conformational changes which changes the enzyme 's activity .
  Fact 0.0: A regulatory site is also known as an allosteric site.
  Fact 0.1: A regulatory site is a site on an allosteric protein.
  Fact 0.2: A modulator molecule binds to a regulatory site on an allosteric protein.
  Fact 0.3: A modulator molecule is also known as an allosteric regulator.
  Fact 0.4: The ligand-binding site on a receptor or enzyme is distinct from the active site.
  Fact 0.5: Allosteric modulators alter enzyme activity by binding to the regulatory site.
  Fact 0.6: Some allosteric regulators activate enzymes.
  Fact 0.7: Other allosteric regulators inhibit enzymes.
  Fact 0.8: The working hypothesis is that when allosteric regulators bind to the regulatory site, the protein undergoes conformational changes.
  Fact 0.9: The working hypothesis is that these conformational changes change the enzyme's activity.

Retrieved items:
<item 1>
ATP synthase
ATP synthase - wikipedia ATP synthase Jump to : navigation , search ATP Synthase Molecular model of ATP synthase determined by X-ray crystallography . Stator is
not shown here . Identifiers EC number 3.6. 3.14 CAS number 9000 - 83 - 3 Databases IntEnz IntEnz view BRENDA BRENDA entry ExPASy NiceZyme view KEGG KEGG entry
MetaCyc metabolic pathway PRIAM profile PDB structures RCSB PDB PDBe PDBsum Gene Ontology AmiGO / QuickGO ( show ) Search PMC articles PubMed articles NCBI
proteins ATP synthase is an enzyme that creates the energy storage molecule adenosine triphosphate ( ATP ) . ATP is the most commonly used `` energy currency ''
of cells for most organisms . It is formed from adenosine diphosphate ( ADP ) and inorganic phosphate ( P ) .
</item 1>

<item 2>
ATP synthase
`` Partial resolution of the enzymes catalyzing oxidative phosphorylation. 8 . Properties of a factor conferring oligomycin sensitivity on mitochondrial
adenosine triphosphatase '' . The Journal of Biological Chemistry . 241 ( 10 ) : 2461 -- 6 . PMID 4223640 . Jump up ^ Mccarty RE ( November 1992 ) . `` A plant
biochemist 's view of H + - ATPases and ATP synthases `` . The Journal of Experimental Biology . 172 ( Pt 1 ) : 431 -- 441 . PMID 9874753 . Jump up ^ PDB : 5ARA
​ ; Zhou A , Rohou A , Schep DG , Bason JV , Montgomery MG , Walker JE , Grigorieff N , Rubinstein JL ( October 2015 ) . `` Structure and conformational states
of the bovine mitochondrial ATP synthase by cryo - EM '' . eLife. 4 : e10180 . doi : 10.7554 / eLife. 10180 . PMC 4718723 . PMID 26439008 .
</item 2>

<item 3>
Graph query row:
entity: regulatory site
relationship: RELATED_TO
related: modulator molecule
</item 3>

<item 4>
Regulatory site
This biochemistry article is a stub . You can help Wikipedia by expanding it . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Regulatory_site&oldid=807494796 '' Categories : Protein structure Biochemistry stubs Hidden categories : Articles to
be merged from October 2017 All articles to be merged Articles lacking sources from December 2009 All articles lacking sources All stub articles Talk About
Wikipedia Add links This page was last edited on 28 October 2017 , at 09 : 38 . About Wikipedia
</item 4>

<item 5>
Regulatory site
Regulatory site - Wikipedia Regulatory site Jump to : navigation , search It has been suggested that this article be merged with Allosteric regulation . (
Discuss ) Proposed since October 2017 . This article does not cite any sources . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed . ( December 2009 ) ( Learn how and when to remove this template message ) A regulatory site ( also known as an
`` allosteric site '' ) is a site on an allosteric protein to which a modulator molecule ( or allosteric regulator ) binds . This ligand - binding site on a
receptor or enzyme is distinct from the active site . Allosteric modulators alter enzyme activity by binding to the regulatory site . Some allosteric regulators
activate enzymes , while others inhibit them . The working hypothesis is that when allosteric regulators bind to the regulatory site , the protein undergoes
conformational changes which changes the enzyme 's activity . This biochemistry article is a stub . You can help Wikipedia by expanding it .
</item 5>

<item 6>
Graph query row:
entity: regulatory site
relationship: RELATED_TO
related: active site
</item 6>

<item 7>
Graph query row:
entity: regulatory site
relationship: LOCATED_IN
related: enzyme
</item 7>

<item 8>
ATP synthase
5.5 - 6 : Polymerization motors Dynamin Tubulin Enzymes Activity Active site Binding site Catalytic triad Oxyanion hole Enzyme promiscuity Catalytically perfect
enzyme Coenzyme Cofactor Enzyme catalysis Regulation Allosteric regulation Cooperativity Enzyme inhibitor Classification EC number Enzyme superfamily Enzyme
family List of enzymes Kinetics Enzyme kinetics Eadie -- Hofstee diagram Hanes -- Woolf plot Lineweaver -- Burk plot Michaelis -- Menten kinetics Types EC1
Oxidoreductases ( list ) EC2 Transferases ( list ) EC3 Hydrolases ( list ) EC4 Lyases ( list ) EC5 Isomerases ( list ) EC6 Ligases ( list ) Molecular and
Cellular Biology portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=ATP_synthase&oldid=821081799 '' Categories : Enzymes Cellular respiration
Photosynthesis EC 3.6.
</item 8>

<item 9>
ATP synthase
The cryo - EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F. Under the right
conditions , the enzyme reaction can also be carried out in reverse , with ATP hydrolysis driving proton pumping across the membrane . The binding change
mechanism involves the active site of a β subunit 's cycling between three states . In the `` loose '' state , ADP and phosphate enter the active site ; in the
adjacent diagram , this is shown in pink . The enzyme then undergoes a change in shape and forces these molecules together , with the active site in the
resulting `` tight '' state ( shown in red ) binding the newly produced ATP molecule with very high affinity . Finally , the active site cycles back to the open
state ( orange ) , releasing ATP and binding more ADP and phosphate , ready for the next cycle of ATP production . Physiological role ( edit ) Like other
enzymes , the activity of F F ATP synthase is reversible .
</item 9>

<item 10>
ATP synthase
`` Atomic model for the dimeric FO region of mitochondrial ATP synthase '' . Science . 358 ( 6365 ) : 936 -- 40 . doi : 10.1126 / science. aao4815 . PMID
29074581 . Jump up ^ Gresser MJ , Myers JA , Boyer PD ( October 1982 ) . `` Catalytic site cooperativity of beef heart mitochondrial F adenosine triphosphatase
. Correlations of initial velocity , bound intermediate , and oxygen exchange measurements with an alternating three - site model '' . The Journal of Biological
Chemistry . 257 ( 20 ) : 12030 -- 8 . PMID 6214554 . Jump up ^ Nakamoto RK , Baylis Scanlon JA , Al - Shawi MK ( August 2008 ) . `` The rotary mechanism of the
ATP synthase '' . Archives of Biochemistry and Biophysics. 476 ( 1 ) : 43 -- 50 . doi : 10.1016 / j. abb. 2008.05. 004 . PMC 2581510 .
</item 10>
=== END ITEM id=0913458771e95c3b ===

=== ITEM id=d6ea022778a41589 ===
Gold sentences and facts:
Sentence 0: A third series was confirmed to be in production by writer Simon Nye at the BFI and Radio Times Television Festival on 8 April 2017 . He described the third series as having `` some exotic new animals '' , and that production would begin in three weeks ' time , upon Keeley Hawes 's arrival in Corfu for filming . It will be set in 1937 .
  Fact 0.0: A third series was confirmed to be in production by writer Simon Nye.
  Fact 0.1: Simon Nye confirmed the third series at the BFI and Radio Times Television Festival on 8 April 2017.
  Fact 0.2: Simon Nye described the third series as having some exotic new animals.
  Fact 0.3: Simon Nye said production of the third series would begin in three weeks' time.
  Fact 0.4: Production of the third series would begin upon Keeley Hawes's arrival in Corfu for filming.
  Fact 0.5: The third series will be set in 1937.

Retrieved items:
<item 1>
The Durrells
See also ( edit ) Gerald Durrell 's Corfu trilogy books : My Family and Other Animals ( 1956 ) Birds , Beasts , and Relatives ( 1969 ) The Garden of the Gods (
1978 ) My Family and Other Animals , a 1987 BBC series based on the Corfu trilogy by Durrell My Family and Other Animals , a 2005 BBC telemovie based on the
1956 book by Durrell References ( edit ) Jump up ^ `` Early life of Gerald Durrell comes to TV screens soon '' . Jersey Evening Post . 24 March 2016 . Retrieved
1 April 2016 . Jump up ^ `` ITV commissions new six - part drama The Durrells '' . ITV Press Centre . 18 June 2015 . Retrieved 1 April 2016 . Jump up ^ `` The
Durrells ( TV Series 2016 -- ) '' . Internet Movie Database . Retrieved 1 April 2016 . Jump up ^ Huntman , Ruth ( 26 March 2016 ) .
</item 1>

<item 2>
The Durrells
12 6 `` Episode 6 '' Edward Hall Simon Nye 28 May 2017 ( 2017 - 05 - 28 ) 5.57 Three different children are born as a fraught Leslie is on hand to help
Lugaretzia 's daughter give birth , Louisa aids with Florence 's , and Gerry is overwhelmed with exuberance at the prospect of some otter cubs - all of which ,
as well as Corfu 's rich culture and wildlife , helps Louisa realise she should refuse to fulfill Hugh 's wish for the two of them to return to England . Hugh ,
however , finds himself in a more dangerous situation , as Vasilia , in a last - ditch attempt to win him back , and in a frantic displeasing at his choosing of
Louisa , stabs him , and flees ; Larry turns up at her doorstep for one last liaison , and she rejects him , adding more disappointment to his day as he later
discovers he has missed everything exciting , an opinion soon put to rest when Donald , prior to leaving Corfu , engages in a fight with Margo 's conceited new
lover , Zoltan .
</item 2>

<item 3>
The Durrells
Characters ( edit ) Main Keeley Hawes as Louisa Durrell Milo Parker as Gerry Durrell Josh O'Connor as Larry Durrell Daisy Waterstone as Margo Durrell Callum
Woodhouse as Leslie Durrell Lucy Black as Florence Petrides Yorgos Karamihos as Dr. Theo Stephanides Alexis Georgoulis as Spiros Halikiopoulos Leslie Caron as
Countess Mavrodaki Ulric von der Esch as Sven Lundblad Anna Savva as Lugaretzia Errika Bigiou as Vasilia ( series 2 ) Recurring James Cosmo as Captain Creech
Maximilian Befort as Max Manolis Emmanouel as Sotos Andrew Bicknell as Headmaster Graham Seed as Mr Trevitt Liz Watts as Nancy Ben Hall as Donald Hara - Joy
Ermidi as Alexia Yorgos Tryfonas as Market Trader Nick Orestis Chaniotakis as Monk Arsenis Grimmas as Guest House Manager Spiros Kasfikis as Durrell 's House
Owner Daniel Lapaine as Hugh Jarvis Series overview ( edit ) Season Episodes Season Premiere Season Finale 6 3 April 2016 ( 2016 -
</item 3>

<item 4>
The Durrells
The series is written by Simon Nye , directed by Steve Barron and Roger Goldby , and produced by Christopher Hall . The executive producers are Lee Morris and
Sally Woodward Gentle. Lee Durrell , Gerald Durrell 's widow and director of the Durrell Wildlife Conservation Trust , acted as consultant . The series is
partly filmed on Corfu and at Ealing Studios in London . Contents ( hide ) 1 Story 2 Characters 3 Series overview 4 Episodes 4.1 Series 1 ( 2016 ) 4.2 Series 2
( 2017 ) 4.3 Series 3 ( 2018 ) 5 Reception 6 Broadcast 7 See also 8 References 9 External links Story ( edit ) The series begins in 1935 , when Louisa Durrell
suddenly announces that she and her four children will move from Bournemouth to the Greek island of Corfu . Her husband has died some years earlier and the
family is experiencing financial problems . A Homeric battle ensues as the family adapts to life on the island which , despite a lack of electricity , is cheap
and an earthly paradise .
</item 4>

<item 5>
The Durrells
Season Episodes Season Premiere Season Finale 6 3 April 2016 ( 2016 - 04 - 03 ) 8 May 2016 ( 2016 - 05 - 08 ) 6 23 April 2017 ( 2017 - 04 - 23 ) 28 May 2017 (
2017 - 05 - 28 ) Episodes ( edit ) Series 1 ( 2016 ) ( edit ) ( 1935 ) No . overall No. in series Title Directed by Written by Original air date UK viewers (
millions ) `` Episode 1 '' Steve Barron Simon Nye 3 April 2016 ( 2016 - 04 - 03 ) 8.81 The island of Corfu 's distinct contrast with England is what enthrals
Bournemouth mother Louisa Durrell to uproot her disenchanted family from a life of no prospects and venture into one with a lessened need for conformity .
However , the appeal becomes ever distant to see upon their arrival .
</item 5>

<item 6>
The Durrells
The Durrells - wikipedia The Durrells Jump to : navigation , search The Durrells Genre Comedy - drama Based on The Corfu Trilogy by Gerald Durrell Written by
Simon Nye Theme music composer Ruth Barrett Country of origin United Kingdom Original language ( s ) English No. of series No. of episodes 12 Production
Executive producer ( s ) Lee Morris Sally Woodward Gentle Simon Nye Producer ( s ) Christopher Hall Cinematography Julian Court , James Aspinall Running time 46
-- 47 minutes Production company ( s ) Sid Gentle Films Masterpiece co-production Release Original network ITV Picture format 16 : 9 1080i Audio format Stereo
Original release 3 April 2016 ( 2016 - 04 - 03 ) -- present External links Website The Durrells ( also known as The Durrells in Corfu on American television )
is a British comedy - drama series based on Gerald Durrell 's three autobiographical books about his family 's four years ( 1935 -- 1939 ) on the Greek Island
of Corfu , which began airing on 3 April 2016 .
</item 6>

<item 7>
The Durrells
10 `` Episode 4 '' Edward Hall Simon Nye 14 May 2017 ( 2017 - 05 - 14 ) 5.96 The Durrells find themselves shut in when a torrential rainstorm hits Corfu , and
not only are the family dealing with the grumpiness emerging from Larry 's affliction with the mumps , but the unusual Mrs. Haddock ( arriving with Aunt
Hermione ) , who claims she is able to be in contact with the dead , and offers to conduct a seance for Louisa to engage with her deceased husband . Donald 's
insistence in conducting his tutoring duties proves annoying to Gerry , while Leslie 's aggravation grows as his indoors reclusion continues , and Margo 's also
, as her acne troubles remain unsolved .
</item 7>

<item 8>
The Durrells
Jump up ^ The episode 's world premiere date was at the BFI and Radio Times Television Festival on 8 April 2017 . Series 3 ( 2018 ) ( edit ) ( 1937 ) A third
series was confirmed to be in production by writer Simon Nye at the BFI and Radio Times Television Festival on 8 April 2017 . He described the third series as
having `` some exotic new animals '' , and that production would begin in three weeks ' time , upon Keeley Hawes 's arrival in Corfu for filming . It will be
set in 1937 . Reception ( edit ) Reception to the first episode was positive , with Gerard O'Donovan ( The Telegraph ) calling it `` a series that 's not only
sun - drenched and liberating , but also catches its source material 's high good humour without labouring it and weaves an authentic sense of the innocent
exoticism of the original , '' before awarding it four stars .
</item 8>

<item 9>
The Durrells
Internet Movie Database . Retrieved 12 August 2016 . Jump up ^ Knox , David ( 30 June 2016 ) . `` Seven picks up UK drama The Durrells '' . TV Tonight .
Retrieved 27 August 2016 . Jump up ^ Dunk , Tiffany ( 24 August 2016 ) . `` Seven takes punt that The Durrells can recreate Downton Abbey 's huge success '' .
news.com.au . Retrieved 27 August 2016 . Jump up ^ `` Screener - TV Listings - `` The Durrells in Corfu on Masterpiece '' `` . 8 October 2016 . Retrieved 8
October 2016 . Jump up ^ `` CBC ANNOUNCES FALL 2017 BROADCAST PREMIERE DATES FOR NEW AND RETURNING TELEVISION SERIES '' . CBC Media Centre . Jump up ^ `` Los
Durrell '' . Retrieved 27 September 2017 . Jump up ^ 28 day data : 7,247,000 on ITV , 1,218,000 on ITV HD and 343,
</item 9>

<item 10>
The Durrells
The show and its episodes attained high ratings on IMDb , around 8 / 10 . BROADCAST ( edit ) Internationally , the series was acquired in Australia by the Seven
Network and premiered on 24 August 2016 . In the United States , PBS began airing the show , retitled as The Durrells in Corfu , on 16 October 2016 at 8pm . The
Durrells started screening in New Zealand on 26 October 2016 on Prime TV . In Canada , the Canadian Broadcasting Corporation began airing Series 1 on 13
September 2017 . In Spain , the series has been acquired by the streaming platform Filmin . A third series of The Durrells was announced by writer Simon Nye at
the BFI and Radio Times Television Festival on 8 April 2017 . It began filming in May 2017 and is expected to air on ITV in the Spring of 2018 .
</item 10>
=== END ITEM id=d6ea022778a41589 ===

=== ITEM id=cd14dc5a90a25b71 ===
Gold sentences and facts:
Sentence 0: ABO and Rh blood type distribution by country ( population averages ) Country Population O+ A+ B+ AB+ O − A − B − AB − Argentina 44,270,440 45.4 % 34.26 % 8.59 % 2.64 % 8.4 % 0.44 % 0.21 % 0.06 % Armenia 2,931,568 29.0 % 46.3 % 12.0 % 5.6 % 2.0 % 3.7 % 1.0 % 0.4 % Australia 24,642,693 40.0 % 31.0 % 8.0 % 2.0 % 9.0 % 7.0 % 2.0 % 1.0 % Austria 8,592,470 30.0 % 33.0 % 12.0 % 6.0 % 7.0 % 8.0 % 3.0 % 1.0 % Bahrain 1,418,695 48.48 % 19.35 % 22.61 % 3.67 % 3.27 % 1.33 % 1.04 % 0.25 % Bangladesh 164,833,667 31.18 % 21.44 % 34.58 % 8.85 % 1.39 % 0.96 % 0.96 % 0.64 % Belgium 11,444,053 38.0 % 34.0 % 8.6 % 4.1 % 7.0 % 6.0 % 1.5 % 0.8 % Bolivia 11,053,376 51.53 % 29.45 % 10.11 % 1.15 % 4.39 % 2.73 % 0.54 % 0.1 % Bosnia and Herzegovina 3,792,730 31.0 % 36.0 % 12.0 % 6.0 % 5.0 % 7.0 % 2.0 % 1.0 % Brazil 211,248,418 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0 % 2.0 % 0.5 % Bulgaria 7,045,097 28.0 % 37.0 % 13.0 % 7.0 % 5.0 % 7.0 % 2.0 % 1.0 % Cambodia 16,077,172 46.7 % 27.2 % 18.5 % 4.9 % 1.3 % 0.8 % 0.5 % 0.1 % Cameroon 24,515,533 42.8 % 38.8 % 12.0 % 3.3 % 1.4 % 1.2 % 0.4 % 0.1 % Canada 36,627,140 39.0 % 36.0 % 7.6 % 2.5 % 7.0 % 6.0 % 1.4 % 0.5 % Chile 18,314,060 85.5 % 8.7 % 3.35 % 1.0 % 1.2 % 0.1 % 0.05 % 0.1 % China 1,388,251,023 47.7 % 27.8 % 18.9 % 5.0 % 0.28 % 0.19 % 0.1 % 0.03 % Colombia 49,069,267 61.3 % 26.11 % 2.28 % 1.47 % 5.13 % 2.7 % 0.7 % 0.31 % Côte d'Ivoire 23,869,656 46.5 % 22.5 % 22.5 % 4.3 % 2.0 % 1.0 % 1.0 % 0.2 % Croatia 4,207,355 29.0 % 36.0 % 15.0 % 5.0 % 5.0 % 6.0 % 3.0 % 1.0 % Cuba 11,486,750 45.8 % 33.5 % 10.2 % 2.9 % 3.6 % 2.8 % 1.0 % 0.2 % Cyprus 1,189,395 35.22 % 40.35 % 11.11 % 4.72 % 3.85 % 3.48 % 0.87 % 0.40 % Czech Republic 10,555,152 27.0 % 36.0 % 15.0 % 7.0 % 5.0 % 6.0 % 3.0 % 1.0 % Denmark 5,711,902 35.0 % 37.0 % 8.0 % 4.0 % 6.0 % 7.0 % 2.0 % 1.0 % Dominican Republic 10,766,932 46.2 % 26.4 % 16.9 % 3.1 % 3.7 % 2.1 % 1.4 % 0.2 % Ecuador 16,665,756 75.0 % 14.0 % 7.1 % 0.5 % 2.38 % 0.7 % 0.3 % 0.02 % Egypt 95,220,838 52.0 % 24.0 % 12.4 % 3.8 % 5.0 % 2.0 % 0.6 % 0.2 % El Salvador 6,171,483 62.0 % 23.0 % 11.0 % 1.0 % 1.0 % 1.0 % 0.7 % 0.3 % Estonia 1,305,745 29.5 % 30.8 % 20.7 % 6.3 % 4.3 % 4.5 % 3.0 % 0.9 % Ethiopia 104,352,727 39.0 % 28.0 % 21.0 % 5.0 % 3.0 % 2.0 % 1.0 % 1.0 % Finland 5,541,328 27.0 % 38.0 % 15.0 % 7.0 % 4.0 % 6.0 % 2.0 % 1.0 % France 64,939,560 36.0 % 37.0 % 9.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Germany 80,635,980 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Greece 10,892,849 37.4 % 32.9 % 11.0 % 3.7 % 7.0 % 5.0 % 2.0 % 1.0 % Guinea 13,291,741 46.88 % 21.64 % 22.86 % 4.52 % 2.0 % 0.9 % 1.0 % 0.2 % Honduras 9,348,898 57.5 % 27.0 % 7.8 % 2.5 % 2.7 % 1.7 % 0.6 % 0.2 % Hong Kong 7,402,115 41.5 % 26.13 % 25.34 % 6.35 % 0.32 % 0.17 % 0.14 % 0.05 % Hungary 9,787,801 27.0 % 33.0 % 16.0 % 8.0 % 5.0 % 7.0 % 3.0 % 1.0 % Iceland 334,311 47.6 % 26.4 % 9.3 % 1.6 % 8.4 % 4.6 % 1.7 % 0.4 % India 1,342,561,902 27.85 % 20.8 % 38.14 % 8.93 % 1.43 % 0.57 % 1.79 % 0.49 % Indonesia 263,519,317 36.82 % 25.87 % 28.85 % 7.96 % 0.18 % 0.13 % 0.15 % 0.04 % Iran 80,982,137 33.5 % 27.0 % 22.2 % 7.0 % 4.0 % 3.0 % 2.5 % 0.8 % Iraq 38,657,787 32.1 % 25.0 % 25.6 % 7.4 % 3.6 % 2.7 % 2.7 % 0.9 % Ireland 4,749,263 47.0 % 26.0 % 9.0 % 2.0 % 8.0 % 5.0 % 2.0 % 1.0 % Israel 8,323,659 32.0 % 34.0 % 17.0 % 7.0 % 3.0 % 4.0 % 2.0 % 1.0 % Italy 59,797,969 39.0 % 36.0 % 7.5 % 2.5 % 7.0 % 6.0 % 1.5 % 0.5 % Jamaica 2,813,316 47.0 % 23.0 % 20.0 % 3.0 % 3.5 % 2.0 % 1.0 % 0.5 % Japan 126,044,340 29.9 % 39.8 % 19.9 % 9.9 % 0.15 % 0.2 % 0.1 % 0.05 % Kenya 48,470,780 45.6 % 25.2 % 21.28 % 4.2 % 1.8 % 1.0 % 0.9 % 0.02 % Latvia 1,944,122 30.6 % 31.0 % 17.0 % 6.0 % 5.4 % 6.0 % 3.0 % 1.0 % Lebanon 6,039,438 38.4 % 32.3 % 9.5 % 3.2 % 7.7 % 6.5 % 1.7 % 0.7 % Libya 6,408,990 42.64 % 28.86 % 11.19 % 4.5 % 6.26 % 4.24 % 1.64 % 0.67 % Lithuania 2,827,573 36.0 % 33.0 % 11.0 % 4.0 % 7.0 % 6.0 % 2.0 % 1.0 % Luxemburg 587,297 35.0 % 37.0 % 9.0 % 4.0 % 6.0 % 6.0 % 2.0 % 1.0 % Macao 606,413 41.5 % 26.1 % 25.4 % 6.3 % 0.33 % 0.17 % 0.15 % 0.05 % Macedonia 2,083,690 30.0 % 34.0 % 15.0 % 6.0 % 5.0 % 6.0 % 3.0 % 1.0 % Malaysia 31,165,480 34.32 % 30.35 % 27.37 % 7.46 % 0.17 % 0.15 % 0.14 % 0.04 % Mauritania 4,266,765 46.3 % 26.68 % 17.47 % 3.85 % 2.8 % 1.6 % 1.1 % 0.2 % Mauritius 1,267,303 38.3 % 26.0 % 25.0 % 6.7 % 1.7 % 1.0 % 1.0 % 0.3 % Mexico 130,227,836 55.79 % 29.9 % 8.0 % 1.63 % 2.7 % 1.5 % 0.4 % 0.08 % Mongolia 3,052,044 55.4 % 27.7 % 10.9 % 4.96 % 0.6 % 0.3 % 0.1 % 0.04 % Myanmar 53,507,932 35.7 % 23.8 % 32.7 % 6.95 % 0.3 % 0.2 % 0.3 % 0.05 % Nepal 29,188,100 35.2 % 28.3 % 27.1 % 8.6 % 0.3 % 0.2 % 0.2 % 0.1 % Netherlands 17,033,012 39.5 % 35.0 % 6.7 % 2.5 % 7.5 % 7.0 % 1.3 % 0.5 % New Zealand 4,604,996 38.0 % 32.0 % 9.0 % 3.0 % 9.0 % 6.0 % 2.0 % 1.0 % Nigeria 191,851,411 51.3 % 22.4 % 20.7 % 2.6 % 1.6 % 0.7 % 0.6 % 0.1 % Norway 5,330,986 33.0 % 41.5 % 6.8 % 3.4 % 6.0 % 7.5 % 1.2 % 0.6 % North Korea 25,432,033 27.15 % 31.08 % 30.15 % 11.32 % 0.08 % 0.1 % 0.1 % 0.03 % Pakistan 180,440,005 24.63 % 20.6 % 34.40 % 9.52 % 4.17 % 2.66 % 3.57 % 0.45 % Papua New Guinea 7,952,511 48.0 % 37.2 % 9.6 % 2.8 % 1.0 % 0.8 % 0.4 % 0.2 % Peru 32,167,717 70.0 % 18.4 % 7.8 % 1.6 % 1.4 % 0.5 % 0.28 % 0.02 % Philippines 103,801,747 36.9 % 28.9 % 26.9 % 6.97 % 0.1 % 0.1 % 0.1 % 0.03 % Poland 38,563,480 31.0 % 32.0 % 15.0 % 7.0 % 6.0 % 6.0 % 2.0 % 1.0 % Portugal 10,264,672 36.3 % 40.0 % 6.6 % 2.9 % 6.0 % 6.6 % 1.1 % 0.5 % Republic of Moldova 4,043,722 28.5 % 31.8 % 17.6 % 7.0 % 5.0 % 6.0 % 3.0 % 1.1 % Romania 19,237,087 28.0 % 37.0 % 14.0 % 7.0 % 5.0 % 6.0 % 2.0 % 1.0 % Russia 143,374,801 28.0 % 30.0 % 20.0 % 7.0 % 4.9 % 5.8 % 3.2 % 1.1 % Saudi Arabia 32,744,532 47.8 % 23.9 % 17.0 % 4.0 % 4.0 % 2.0 % 1.0 % 0.3 % Serbia 8,776,827 31.92 % 35.28 % 12.6 % 4.2 % 6.08 % 6.72 % 2.4 % 0.8 % Singapore 5,784,819 43.6 % 23.9 % 24.4 % 6.0 % 0.9 % 0.7 % 0.4 % 0.1 % Slovakia 5,432,265 27.2 % 35.7 % 15.3 % 6.8 % 4.8 % 6.3 % 2.7 % 1.2 % Slovenia 2,071,258 31.0 % 33.0 % 12.0 % 6.0 % 7.0 % 7.0 % 3.0 % 1.0 % South Africa 55,437,815 39.0 % 32.0 % 12.0 % 3.0 % 6.0 % 5.0 % 2.0 % 1.0 % South Korea 50,748,307 27.9 % 33.87 % 26.92 % 10.98 % 0.1 % 0.13 % 0.08 % 0.02 % Spain 46,070,163 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0 % 2.0 % 0.5 % Sudan 40,827,975 48.0 % 27.7 % 15.2 % 2.8 % 3.5 % 1.8 % 0.8 % 0.2 % Sweden 9,920,843 32.0 % 37.0 % 10.0 % 5.0 % 6.0 % 7.0 % 2.0 % 1.0 % Switzerland 8,454,321 35.0 % 40.0 % 7.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Syria 18,920,698 43.0 % 30.0 % 14.0 % 3.7 % 5.0 % 3.0 % 1.0 % 0.3 % Taiwan 23,234,936 43.9 % 25.9 % 23.9 % 6.0 % 0.17 % 0.1 % 0.01 % 0.02 % Thailand 68,298,027 40.8 % 16.9 % 36.8 % 4.97 % 0.2 % 0.1 % 0.2 % 0.03 % Turkey 80,420,065 29.8 % 37.8 % 14.2 % 7.2 % 3.9 % 4.7 % 1.6 % 0.8 % Uganda 43,276,492 43.7 % 39.0 % 10.7 % 3.9 % 1.3 % 1.0 % 0.3 % 0.1 % Ukraine 44,404,078 32.0 % 34.0 % 15.0 % 5.0 % 5.0 % 6.0 % 2.0 % 1.0 % United Arab Emirates 3,398,017 44.1 % 21.9 % 20.9 % 4.3 % 4.3 % 2.1 % 2.0 % 0.4 % United Kingdom 65,512,375 37.0 % 35.0 % 8.0 % 3.0 % 7.0 % 7.0 % 2.0 % 1.0 % United States 326,481,533 37.4 % 35.7 % 8.5 % 3.4 % 6.6 % 6.3 % 1.5 % 0.6 % Venezuela 32,011,082 58.5 % 28.3 % 5.3 % 1.8 % 3.5 % 1.7 % 0.7 % 0.2 % Vietnam 95,414,640 41.7 % 21.9 % 30.8 % 4.98 % 0.3 % 0.1 % 0.2 % 0.02 % Zimbabwe 16,642,880 63.3 % 18.5 % 14.6 % 0.99 % 1.7 % 0.5 % 0.4 % 0.01 % World 6,635,281,257 38.67 % 27.42 % 22.02 % 5.88 % 2.55 % 1.99 % 1.11 % 0.36 %
  Fact 0.0: The table gives ABO and Rh blood type distribution by country as population averages.
  Fact 0.1: Each row lists a country, its population, and the percentages of O+, A+, B+, AB+, O−, A−, B−, and AB− blood types.
  Fact 0.2: Argentina has a population of 44,270,440, with blood type percentages O+ 45.4, A+ 34.26, B+ 8.59, AB+ 2.64, O− 8.4, A− 0.44, B− 0.21, AB− 0.06.
  Fact 0.3: Armenia has a population of 2,931,568, with blood type percentages O+ 29.0, A+ 46.3, B+ 12.0, AB+ 5.6, O− 2.0, A− 3.7, B− 1.0, AB− 0.4.
  Fact 0.4: Australia has a population of 24,642,693, with blood type percentages O+ 40.0, A+ 31.0, B+ 8.0, AB+ 2.0, O− 9.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.5: Austria has a population of 8,592,470, with blood type percentages O+ 30.0, A+ 33.0, B+ 12.0, AB+ 6.0, O− 7.0, A− 8.0, B− 3.0, AB− 1.0.
  Fact 0.6: Bahrain has a population of 1,418,695, with blood type percentages O+ 48.48, A+ 19.35, B+ 22.61, AB+ 3.67, O− 3.27, A− 1.33, B− 1.04, AB− 0.25.
  Fact 0.7: Bangladesh has a population of 164,833,667, with blood type percentages O+ 31.18, A+ 21.44, B+ 34.58, AB+ 8.85, O− 1.39, A− 0.96, B− 0.96, AB− 0.64.
  Fact 0.8: Belgium has a population of 11,444,053, with blood type percentages O+ 38.0, A+ 34.0, B+ 8.6, AB+ 4.1, O− 7.0, A− 6.0, B− 1.5, AB− 0.8.
  Fact 0.9: Bolivia has a population of 11,053,376, with blood type percentages O+ 51.53, A+ 29.45, B+ 10.11, AB+ 1.15, O− 4.39, A− 2.73, B− 0.54, AB− 0.1.
  Fact 0.10: Bosnia and Herzegovina has a population of 3,792,730, with blood type percentages O+ 31.0, A+ 36.0, B+ 12.0, AB+ 6.0, O− 5.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.11: Brazil has a population of 211,248,418, with blood type percentages O+ 36.0, A+ 34.0, B+ 8.0, AB+ 2.5, O− 9.0, A− 8.0, B− 2.0, AB− 0.5.
  Fact 0.12: Bulgaria has a population of 7,045,097, with blood type percentages O+ 28.0, A+ 37.0, B+ 13.0, AB+ 7.0, O− 5.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.13: Cambodia has a population of 16,077,172, with blood type percentages O+ 46.7, A+ 27.2, B+ 18.5, AB+ 4.9, O− 1.3, A− 0.8, B− 0.5, AB− 0.1.
  Fact 0.14: Cameroon has a population of 24,515,533, with blood type percentages O+ 42.8, A+ 38.8, B+ 12.0, AB+ 3.3, O− 1.4, A− 1.2, B− 0.4, AB− 0.1.
  Fact 0.15: Canada has a population of 36,627,140, with blood type percentages O+ 39.0, A+ 36.0, B+ 7.6, AB+ 2.5, O− 7.0, A− 6.0, B− 1.4, AB− 0.5.
  Fact 0.16: Chile has a population of 18,314,060, with blood type percentages O+ 85.5, A+ 8.7, B+ 3.35, AB+ 1.0, O− 1.2, A− 0.1, B− 0.05, AB− 0.1.
  Fact 0.17: China has a population of 1,388,251,023, with blood type percentages O+ 47.7, A+ 27.8, B+ 18.9, AB+ 5.0, O− 0.28, A− 0.19, B− 0.1, AB− 0.03.
  Fact 0.18: Colombia has a population of 49,069,267, with blood type percentages O+ 61.3, A+ 26.11, B+ 2.28, AB+ 1.47, O− 5.13, A− 2.7, B− 0.7, AB− 0.31.
  Fact 0.19: Côte d'Ivoire has a population of 23,869,656, with blood type percentages O+ 46.5, A+ 22.5, B+ 22.5, AB+ 4.3, O− 2.0, A− 1.0, B− 1.0, AB− 0.2.
  Fact 0.20: Croatia has a population of 4,207,355, with blood type percentages O+ 29.0, A+ 36.0, B+ 15.0, AB+ 5.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.21: Cuba has a population of 11,486,750, with blood type percentages O+ 45.8, A+ 33.5, B+ 10.2, AB+ 2.9, O− 3.6, A− 2.8, B− 1.0, AB− 0.2.
  Fact 0.22: Cyprus has a population of 1,189,395, with blood type percentages O+ 35.22, A+ 40.35, B+ 11.11, AB+ 4.72, O− 3.85, A− 3.48, B− 0.87, AB− 0.40.
  Fact 0.23: Czech Republic has a population of 10,555,152, with blood type percentages O+ 27.0, A+ 36.0, B+ 15.0, AB+ 7.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.24: Denmark has a population of 5,711,902, with blood type percentages O+ 35.0, A+ 37.0, B+ 8.0, AB+ 4.0, O− 6.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.25: Dominican Republic has a population of 10,766,932, with blood type percentages O+ 46.2, A+ 26.4, B+ 16.9, AB+ 3.1, O− 3.7, A− 2.1, B− 1.4, AB− 0.2.
  Fact 0.26: Ecuador has a population of 16,665,756, with blood type percentages O+ 75.0, A+ 14.0, B+ 7.1, AB+ 0.5, O− 2.38, A− 0.7, B− 0.3, AB− 0.02.
  Fact 0.27: Egypt has a population of 95,220,838, with blood type percentages O+ 52.0, A+ 24.0, B+ 12.4, AB+ 3.8, O− 5.0, A− 2.0, B− 0.6, AB− 0.2.
  Fact 0.28: El Salvador has a population of 6,171,483, with blood type percentages O+ 62.0, A+ 23.0, B+ 11.0, AB+ 1.0, O− 1.0, A− 1.0, B− 0.7, AB− 0.3.
  Fact 0.29: Estonia has a population of 1,305,745, with blood type percentages O+ 29.5, A+ 30.8, B+ 20.7, AB+ 6.3, O− 4.3, A− 4.5, B− 3.0, AB− 0.9.
  Fact 0.30: Ethiopia has a population of 104,352,727, with blood type percentages O+ 39.0, A+ 28.0, B+ 21.0, AB+ 5.0, O− 3.0, A− 2.0, B− 1.0, AB− 1.0.
  Fact 0.31: Finland has a population of 5,541,328, with blood type percentages O+ 27.0, A+ 38.0, B+ 15.0, AB+ 7.0, O− 4.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.32: France has a population of 64,939,560, with blood type percentages O+ 36.0, A+ 37.0, B+ 9.0, AB+ 3.0, O− 6.0, A− 7.0, B− 1.0, AB− 1.0.
  Fact 0.33: Germany has a population of 80,635,980, with blood type percentages O+ 35.0, A+ 37.0, B+ 9.0, AB+ 4.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.34: Greece has a population of 10,892,849, with blood type percentages O+ 37.4, A+ 32.9, B+ 11.0, AB+ 3.7, O− 7.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.35: Guinea has a population of 13,291,741, with blood type percentages O+ 46.88, A+ 21.64, B+ 22.86, AB+ 4.52, O− 2.0, A− 0.9, B− 1.0, AB− 0.2.
  Fact 0.36: Honduras has a population of 9,348,898, with blood type percentages O+ 57.5, A+ 27.0, B+ 7.8, AB+ 2.5, O− 2.7, A− 1.7, B− 0.6, AB− 0.2.
  Fact 0.37: Hong Kong has a population of 7,402,115, with blood type percentages O+ 41.5, A+ 26.13, B+ 25.34, AB+ 6.35, O− 0.32, A− 0.17, B− 0.14, AB− 0.05.
  Fact 0.38: Hungary has a population of 9,787,801, with blood type percentages O+ 27.0, A+ 33.0, B+ 16.0, AB+ 8.0, O− 5.0, A− 7.0, B− 3.0, AB− 1.0.
  Fact 0.39: Iceland has a population of 334,311, with blood type percentages O+ 47.6, A+ 26.4, B+ 9.3, AB+ 1.6, O− 8.4, A− 4.6, B− 1.7, AB− 0.4.
  Fact 0.40: India has a population of 1,342,561,902, with blood type percentages O+ 27.85, A+ 20.8, B+ 38.14, AB+ 8.93, O− 1.43, A− 0.57, B− 1.79, AB− 0.49.
  Fact 0.41: Indonesia has a population of 263,519,317, with blood type percentages O+ 36.82, A+ 25.87, B+ 28.85, AB+ 7.96, O− 0.18, A− 0.13, B− 0.15, AB− 0.04.
  Fact 0.42: Iran has a population of 80,982,137, with blood type percentages O+ 33.5, A+ 27.0, B+ 22.2, AB+ 7.0, O− 4.0, A− 3.0, B− 2.5, AB− 0.8.
  Fact 0.43: Iraq has a population of 38,657,787, with blood type percentages O+ 32.1, A+ 25.0, B+ 25.6, AB+ 7.4, O− 3.6, A− 2.7, B− 2.7, AB− 0.9.
  Fact 0.44: Ireland has a population of 4,749,263, with blood type percentages O+ 47.0, A+ 26.0, B+ 9.0, AB+ 2.0, O− 8.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.45: Israel has a population of 8,323,659, with blood type percentages O+ 32.0, A+ 34.0, B+ 17.0, AB+ 7.0, O− 3.0, A− 4.0, B− 2.0, AB− 1.0.
  Fact 0.46: Italy has a population of 59,797,969, with blood type percentages O+ 39.0, A+ 36.0, B+ 7.5, AB+ 2.5, O− 7.0, A− 6.0, B− 1.5, AB− 0.5.
  Fact 0.47: Jamaica has a population of 2,813,316, with blood type percentages O+ 47.0, A+ 23.0, B+ 20.0, AB+ 3.0, O− 3.5, A− 2.0, B− 1.0, AB− 0.5.
  Fact 0.48: Japan has a population of 126,044,340, with blood type percentages O+ 29.9, A+ 39.8, B+ 19.9, AB+ 9.9, O− 0.15, A− 0.2, B− 0.1, AB− 0.05.
  Fact 0.49: Kenya has a population of 48,470,780, with blood type percentages O+ 45.6, A+ 25.2, B+ 21.28, AB+ 4.2, O− 1.8, A− 1.0, B− 0.9, AB− 0.02.
  Fact 0.50: Latvia has a population of 1,944,122, with blood type percentages O+ 30.6, A+ 31.0, B+ 17.0, AB+ 6.0, O− 5.4, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.51: Lebanon has a population of 6,039,438, with blood type percentages O+ 38.4, A+ 32.3, B+ 9.5, AB+ 3.2, O− 7.7, A− 6.5, B− 1.7, AB− 0.7.
  Fact 0.52: Libya has a population of 6,408,990, with blood type percentages O+ 42.64, A+ 28.86, B+ 11.19, AB+ 4.5, O− 6.26, A− 4.24, B− 1.64, AB− 0.67.
  Fact 0.53: Lithuania has a population of 2,827,573, with blood type percentages O+ 36.0, A+ 33.0, B+ 11.0, AB+ 4.0, O− 7.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.54: Luxemburg has a population of 587,297, with blood type percentages O+ 35.0, A+ 37.0, B+ 9.0, AB+ 4.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.55: Macao has a population of 606,413, with blood type percentages O+ 41.5, A+ 26.1, B+ 25.4, AB+ 6.3, O− 0.33, A− 0.17, B− 0.15, AB− 0.05.
  Fact 0.56: Macedonia has a population of 2,083,690, with blood type percentages O+ 30.0, A+ 34.0, B+ 15.0, AB+ 6.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.0.
  Fact 0.57: Malaysia has a population of 31,165,480, with blood type percentages O+ 34.32, A+ 30.35, B+ 27.37, AB+ 7.46, O− 0.17, A− 0.15, B− 0.14, AB− 0.04.
  Fact 0.58: Mauritania has a population of 4,266,765, with blood type percentages O+ 46.3, A+ 26.68, B+ 17.47, AB+ 3.85, O− 2.8, A− 1.6, B− 1.1, AB− 0.2.
  Fact 0.59: Mauritius has a population of 1,267,303, with blood type percentages O+ 38.3, A+ 26.0, B+ 25.0, AB+ 6.7, O− 1.7, A− 1.0, B− 1.0, AB− 0.3.
  Fact 0.60: Mexico has a population of 130,227,836, with blood type percentages O+ 55.79, A+ 29.9, B+ 8.0, AB+ 1.63, O− 2.7, A− 1.5, B− 0.4, AB− 0.08.
  Fact 0.61: Mongolia has a population of 3,052,044, with blood type percentages O+ 55.4, A+ 27.7, B+ 10.9, AB+ 4.96, O− 0.6, A− 0.3, B− 0.1, AB− 0.04.
  Fact 0.62: Myanmar has a population of 53,507,932, with blood type percentages O+ 35.7, A+ 23.8, B+ 32.7, AB+ 6.95, O− 0.3, A− 0.2, B− 0.3, AB− 0.05.
  Fact 0.63: Nepal has a population of 29,188,100, with blood type percentages O+ 35.2, A+ 28.3, B+ 27.1, AB+ 8.6, O− 0.3, A− 0.2, B− 0.2, AB− 0.1.
  Fact 0.64: Netherlands has a population of 17,033,012, with blood type percentages O+ 39.5, A+ 35.0, B+ 6.7, AB+ 2.5, O− 7.5, A− 7.0, B− 1.3, AB− 0.5.
  Fact 0.65: New Zealand has a population of 4,604,996, with blood type percentages O+ 38.0, A+ 32.0, B+ 9.0, AB+ 3.0, O− 9.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.66: Nigeria has a population of 191,851,411, with blood type percentages O+ 51.3, A+ 22.4, B+ 20.7, AB+ 2.6, O− 1.6, A− 0.7, B− 0.6, AB− 0.1.
  Fact 0.67: Norway has a population of 5,330,986, with blood type percentages O+ 33.0, A+ 41.5, B+ 6.8, AB+ 3.4, O− 6.0, A− 7.5, B− 1.2, AB− 0.6.
  Fact 0.68: North Korea has a population of 25,432,033, with blood type percentages O+ 27.15, A+ 31.08, B+ 30.15, AB+ 11.32, O− 0.08, A− 0.1, B− 0.1, AB− 0.03.
  Fact 0.69: Pakistan has a population of 180,440,005, with blood type percentages O+ 24.63, A+ 20.6, B+ 34.40, AB+ 9.52, O− 4.17, A− 2.66, B− 3.57, AB− 0.45.
  Fact 0.70: Papua New Guinea has a population of 7,952,511, with blood type percentages O+ 48.0, A+ 37.2, B+ 9.6, AB+ 2.8, O− 1.0, A− 0.8, B− 0.4, AB− 0.2.
  Fact 0.71: Peru has a population of 32,167,717, with blood type percentages O+ 70.0, A+ 18.4, B+ 7.8, AB+ 1.6, O− 1.4, A− 0.5, B− 0.28, AB− 0.02.
  Fact 0.72: Philippines has a population of 103,801,747, with blood type percentages O+ 36.9, A+ 28.9, B+ 26.9, AB+ 6.97, O− 0.1, A− 0.1, B− 0.1, AB− 0.03.
  Fact 0.73: Poland has a population of 38,563,480, with blood type percentages O+ 31.0, A+ 32.0, B+ 15.0, AB+ 7.0, O− 6.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.74: Portugal has a population of 10,264,672, with blood type percentages O+ 36.3, A+ 40.0, B+ 6.6, AB+ 2.9, O− 6.0, A− 6.6, B− 1.1, AB− 0.5.
  Fact 0.75: Republic of Moldova has a population of 4,043,722, with blood type percentages O+ 28.5, A+ 31.8, B+ 17.6, AB+ 7.0, O− 5.0, A− 6.0, B− 3.0, AB− 1.1.
  Fact 0.76: Romania has a population of 19,237,087, with blood type percentages O+ 28.0, A+ 37.0, B+ 14.0, AB+ 7.0, O− 5.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.77: Russia has a population of 143,374,801, with blood type percentages O+ 28.0, A+ 30.0, B+ 20.0, AB+ 7.0, O− 4.9, A− 5.8, B− 3.2, AB− 1.1.
  Fact 0.78: Saudi Arabia has a population of 32,744,532, with blood type percentages O+ 47.8, A+ 23.9, B+ 17.0, AB+ 4.0, O− 4.0, A− 2.0, B− 1.0, AB− 0.3.
  Fact 0.79: Serbia has a population of 8,776,827, with blood type percentages O+ 31.92, A+ 35.28, B+ 12.6, AB+ 4.2, O− 6.08, A− 6.72, B− 2.4, AB− 0.8.
  Fact 0.80: Singapore has a population of 5,784,819, with blood type percentages O+ 43.6, A+ 23.9, B+ 24.4, AB+ 6.0, O− 0.9, A− 0.7, B− 0.4, AB− 0.1.
  Fact 0.81: Slovakia has a population of 5,432,265, with blood type percentages O+ 27.2, A+ 35.7, B+ 15.3, AB+ 6.8, O− 4.8, A− 6.3, B− 2.7, AB− 1.2.
  Fact 0.82: Slovenia has a population of 2,071,258, with blood type percentages O+ 31.0, A+ 33.0, B+ 12.0, AB+ 6.0, O− 7.0, A− 7.0, B− 3.0, AB− 1.0.
  Fact 0.83: South Africa has a population of 55,437,815, with blood type percentages O+ 39.0, A+ 32.0, B+ 12.0, AB+ 3.0, O− 6.0, A− 5.0, B− 2.0, AB− 1.0.
  Fact 0.84: South Korea has a population of 50,748,307, with blood type percentages O+ 27.9, A+ 33.87, B+ 26.92, AB+ 10.98, O− 0.1, A− 0.13, B− 0.08, AB− 0.02.
  Fact 0.85: Spain has a population of 46,070,163, with blood type percentages O+ 36.0, A+ 34.0, B+ 8.0, AB+ 2.5, O− 9.0, A− 8.0, B− 2.0, AB− 0.5.
  Fact 0.86: Sudan has a population of 40,827,975, with blood type percentages O+ 48.0, A+ 27.7, B+ 15.2, AB+ 2.8, O− 3.5, A− 1.8, B− 0.8, AB− 0.2.
  Fact 0.87: Sweden has a population of 9,920,843, with blood type percentages O+ 32.0, A+ 37.0, B+ 10.0, AB+ 5.0, O− 6.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.88: Switzerland has a population of 8,454,321, with blood type percentages O+ 35.0, A+ 40.0, B+ 7.0, AB+ 3.0, O− 6.0, A− 7.0, B− 1.0, AB− 1.0.
  Fact 0.89: Syria has a population of 18,920,698, with blood type percentages O+ 43.0, A+ 30.0, B+ 14.0, AB+ 3.7, O− 5.0, A− 3.0, B− 1.0, AB− 0.3.
  Fact 0.90: Taiwan has a population of 23,234,936, with blood type percentages O+ 43.9, A+ 25.9, B+ 23.9, AB+ 6.0, O− 0.17, A− 0.1, B− 0.01, AB− 0.02.
  Fact 0.91: Thailand has a population of 68,298,027, with blood type percentages O+ 40.8, A+ 16.9, B+ 36.8, AB+ 4.97, O− 0.2, A− 0.1, B− 0.2, AB− 0.03.
  Fact 0.92: Turkey has a population of 80,420,065, with blood type percentages O+ 29.8, A+ 37.8, B+ 14.2, AB+ 7.2, O− 3.9, A− 4.7, B− 1.6, AB− 0.8.
  Fact 0.93: Uganda has a population of 43,276,492, with blood type percentages O+ 43.7, A+ 39.0, B+ 10.7, AB+ 3.9, O− 1.3, A− 1.0, B− 0.3, AB− 0.1.
  Fact 0.94: Ukraine has a population of 44,404,078, with blood type percentages O+ 32.0, A+ 34.0, B+ 15.0, AB+ 5.0, O− 5.0, A− 6.0, B− 2.0, AB− 1.0.
  Fact 0.95: United Arab Emirates has a population of 3,398,017, with blood type percentages O+ 44.1, A+ 21.9, B+ 20.9, AB+ 4.3, O− 4.3, A− 2.1, B− 2.0, AB− 0.4.
  Fact 0.96: United Kingdom has a population of 65,512,375, with blood type percentages O+ 37.0, A+ 35.0, B+ 8.0, AB+ 3.0, O− 7.0, A− 7.0, B− 2.0, AB− 1.0.
  Fact 0.97: United States has a population of 326,481,533, with blood type percentages O+ 37.4, A+ 35.7, B+ 8.5, AB+ 3.4, O− 6.6, A− 6.3, B− 1.5, AB− 0.6.
  Fact 0.98: Venezuela has a population of 32,011,082, with blood type percentages O+ 58.5, A+ 28.3, B+ 5.3, AB+ 1.8, O− 3.5, A− 1.7, B− 0.7, AB− 0.2.
  Fact 0.99: Vietnam has a population of 95,414,640, with blood type percentages O+ 41.7, A+ 21.9, B+ 30.8, AB+ 4.98, O− 0.3, A− 0.1, B− 0.2, AB− 0.02.
  Fact 0.100: Zimbabwe has a population of 16,642,880, with blood type percentages O+ 63.3, A+ 18.5, B+ 14.6, AB+ 0.99, O− 1.7, A− 0.5, B− 0.4, AB− 0.01.
  Fact 0.101: The World row has a population of 6,635,281,257, with blood type percentages O+ 38.67, A+ 27.42, B+ 22.02, AB+ 5.88, O− 2.55, A− 1.99, B− 1.11, AB− 0.36.

Retrieved items:
<item 1>
Blood type distribution by country
975 48.0 % 27.7 % 15.2 % 2.8 % 3.5 % 1.8 % 0.8 % 0.2 % Sweden 9,920,843 32.0 % 37.0 % 10.0 % 5.0 % 6.0 % 7.0 % 2.0 % 1.0 % Switzerland 8,454,321 35.0 % 40.0 %
7.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Syria 18,920,698 43.0 % 30.0 % 14.0 % 3.7 % 5.0 % 3.0 % 1.0 % 0.3 % Taiwan 23,234,936 43.9 % 25.9 % 23.9 % 6.0 % 0.17 % 0.1
% 0.01 % 0.02 % Thailand 68,298,027 40.8 % 16.9 % 36.
</item 1>

<item 2>
Blood type distribution by country
40 11 Spanish 38 47 10 5 Sudanese 62 16 21 0 Swedish 38 47 10 5 Swiss 40 50 7 Tatars 28 30 29 13 Thais 37 22 33 8 Turks 43 34 18 6 Ukrainians 37 40 18 6 Black
Americans 49 27 20 White Americans 45 40 11 Vietnamese 42 22 30 5 Mean 43.91 34.80 16.55 5.14 Standard deviation 16.87 13.80 9.97 3.41 Blood group B has its
highest frequency in South Asia where it ranks first as the largest share of the earth 's population . In Southeast Asia the share of the population is high ,
especially in Thailand and Indonesia , then in East Asia , Northern Asia and neighboring Central Asia , and its incidence diminishes both towards the east and
the west , falling to single digit percentages in Netherlands , Norway , Portugal and Switzerland . It is believed to have been entirely absent from Native
American and Australian Aboriginal populations prior to the arrival of Europeans in those areas .
</item 2>

<item 3>
Blood type distribution by country
Retrieved 2010 - 11 - 19 . Jump up ^ `` Federación Nacional de Donantes de Sangre / La sangre / Grupos '' . Donantesdesangre.net . Retrieved 2010 - 11 - 19 .
Jump up ^ `` Frequency of major blood groups in the Swedish population '' . Geblod.nu. 2007 - 10 - 02 . Archived from the original on 24 November 2010 .
Retrieved 2010 - 11 - 19 . Jump up ^ - BLUTSPENDE https://www.blutspende.ch/fr/don_de_sang/sanguin_groupe/groupe_sanguin - BLUTSPENDE Check url = value ( help )
. Missing or empty title = ( help ) Jump up ^ `` Turkey Blood Group Site '' . Kangrubu.com . Archived from the original on 29 May 2010 . Retrieved 2010 - 11 -
19 . Jump up ^ `` Blood Group Basics '' . Blood.co.uk . Retrieved 2017 - 07 - 30 .
</item 3>

<item 4>
Blood type distribution by country
7 % 15.3 % 6.8 % 4.8 % 6.3 % 2.7 % 1.2 % Slovenia 2,071,258 31.0 % 33.0 % 12.0 % 6.0 % 7.0 % 7.0 % 3.0 % 1.0 % South Africa 55,437,815 39.0 % 32.0 % 12.0 % 3.0
% 6.0 % 5.0 % 2.0 % 1.0 % South Korea 50,748,307 27.9 % 33.87 % 26.92 % 10.98 % 0.1 % 0.13 % 0.08 % 0.02 % Spain 46,070,163 36.0 % 34.0 % 8.0 % 2.5 % 9.0 % 8.0
% 2.0 % 0.5 % Sudan 40,827,975 48.0 % 27.7 % 15.2 % 2.
</item 4>

<item 5>
Blood type distribution by country
2 % El Salvador 6,171,483 62.0 % 23.0 % 11.0 % 1.0 % 1.0 % 1.0 % 0.7 % 0.3 % Estonia 1,305,745 29.5 % 30.8 % 20.7 % 6.3 % 4.3 % 4.5 % 3.0 % 0.9 % Ethiopia
104,352,727 39.0 % 28.0 % 21.0 % 5.0 % 3.0 % 2.0 % 1.0 % 1.0 % Finland 5,541,328 27.0 % 38.0 % 15.0 % 7.0 % 4.0 % 6.0 % 2.0 % 1.0 % France 64,939,560 36.0 %
37.0 % 9.0 % 3.0 % 6.0 % 7.0 % 1.0 % 1.0 % Germany 80,635,980 35.
</item 5>

<item 6>
Blood type distribution by country
Blood group A is associated with high frequencies in Europe , especially in Scandinavia and Central Europe , although its highest frequencies occur in some
Australian Aboriginal populations and the Blackfoot Indians of Montana , the US . Maps of allele ABO among native populations ( edit ) Map of allele A among
native populations Map of allele B among native populations Map of allele O among native populations References ( edit ) Jump up ^ CIA World Factbook Jump up ^
`` Study of Blood Immunogenetical Markers in Armenian Population '' ( PDF ) . 1989 . Jump up ^ Blood Types - What Are They ? , Australian Red Cross Jump up ^ ``
Austrian Red Cross - Blood Donor Information '' . Old.roteskreuz.at. 2006 - 03 - 21 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Rode Kruis Wielsbeke - Blood Donor
information material '' . Rodekruiswielsbeke.be . Archived from the original on 26 November 2010 . Retrieved 2010 - 11 - 19 .
</item 6>

<item 7>
Blood type distribution by country
0 % 5.0 % 6.0 % 2.0 % 1.0 % Russia 143,374,801 28.0 % 30.0 % 20.0 % 7.0 % 4.9 % 5.8 % 3.2 % 1.1 % Saudi Arabia 32,744,532 47.8 % 23.9 % 17.0 % 4.0 % 4.0 % 2.0 %
1.0 % 0.3 % Serbia 8,776,827 31.92 % 35.28 % 12.6 % 4.2 % 6.08 % 6.72 % 2.4 % 0.8 % Singapore 5,784,819 43.6 % 23.9 % 24.4 % 6.0 % 0.9 % 0.7 % 0.4 % 0.1 %
Slovakia 5,432,265 27.2 % 35.7 % 15.3 % 6.8 % 4.8 % 6.
</item 7>

<item 8>
Blood type distribution by country
Retrieved 2017 - 11 - 23 . Jump up ^ `` Veregruppidest '' ( Blood groups ) . verekeskus.ee ( in Estonian ) . Retrieved 2017 - 11 - 23 . Jump up ^ ``
Suomalaisten veriryhmäjakauma '' ( in Finnish ) . 2009 - 08 - 21 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Les groupes sanguins ( système ABO ) '' . Centre
Hospitalier Princesse GRACE - Monaco ( in French ) . C.H.P.G. MONACO. 2005 . Retrieved 2008 - 07 - 15 . Jump up ^ `` Blóðflokkar '' ( in Icelandic ) ...
landspitali.is . Archived from the original on 2011 - 07 - 19 . Retrieved 2010 - 11 - 19 . Jump up ^ `` Blood Type Frequencies by Country including the Rh
Factor '' .
</item 8>

<item 9>
Blood type distribution by country
Jump up ^ `` Portuguese Blood Institute '' ( in Portuguese ) . ( assuming Rh and AB antigens are independent ) Jump up ^ Sanguina Jump up ^ BASHAWRI , Layla A.
; AL - MULHIMiz , Abdul Az ; AHMAD , Malik S. ( 2001 ) . `` Frequency of ABO blood groups in the eastern region of Saudi Arabia '' . Saudi Medical Journal . 22
( 11 ) : 1008 -- 1101 . ISSN 0379 - 5284 . PMID 11744976 . Jump up ^ `` Najčastejšie otázky '' ( Frequently Asked Questions ) ( in Slovak ) . Národná transfúzna
služba SR . Retrieved 2017 - 09 - 22 . Jump up ^ `` What 's Your Type ? '' . South African National Blood Service . Archived from the original on 2010 - 02 - 06
. Retrieved 2010 - 11 - 19 .
</item 9>

<item 10>
Blood type distribution by country
In countries such as Austria , Denmark , Norway , and Switzerland , about 45 -- 50 % of the population have this blood type , whereas about 40 % of Poles and
Ukrainians do so . The highest frequencies are found in small , unrelated populations . For example , about 80 % of the Blackfoot Indians of Montana have blood
type A ... Jump up ^ Technical Monograph No. 2 : The ABO Blood Group System and ABO Subgroups ( PDF ) . Biotec . March 2005 . Archived from the original ( PDF )
on 6 February 2007 ... The frequency of blood group A is quite high ( 25 -- 55 % ) in Europe , especially in Scandinavia and parts of central Europe . High
group A frequency is also found in the Aboriginal people of South Australia ( up to 45 % ) and in certain American Indian tribes where the frequency reaches 35
% ...
</item 10>
=== END ITEM id=cd14dc5a90a25b71 ===

=== ITEM id=7f13f56cf1ac55e4 ===
Gold sentences and facts:
Sentence 0: The coastline of the island consists of a combination of long beaches , dunes , red sandstone cliffs , salt water marshes and numerous bays and harbours . The beaches , dunes and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration which oxidizes upon exposure to the air . The geological properties of a white silica sand found at Basin Head are unique in the province ; the sand grains cause a scrubbing noise as they rub against each other when walked on . Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours . The magnificent sand dunes at Greenwich are of particular significance . The shifting , parabolic dune system is home to a variety of birds and rare plants and is also a site of significant archeological interest .
  Fact 0.0: The coastline of the island consists of a combination of long beaches, dunes, red sandstone cliffs, salt water marshes, and numerous bays and harbours.
  Fact 0.1: The beaches, dunes, and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration.
  Fact 0.2: That iron concentration oxidizes upon exposure to air.
  Fact 0.3: The geological properties of a white silica sand found at Basin Head are unique in the province.
  Fact 0.4: The sand grains at Basin Head cause a scrubbing noise as they rub against each other when walked on.
  Fact 0.5: Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours.
  Fact 0.6: The magnificent sand dunes at Greenwich are of particular significance.
  Fact 0.7: The shifting, parabolic dune system at Greenwich is home to a variety of birds and rare plants.
  Fact 0.8: The shifting, parabolic dune system at Greenwich is also a site of significant archeological interest.

Retrieved items:
<item 1>
Geography of Prince Edward Island
Geography of Prince Edward Island - wikipedia Geography of Prince Edward Island Jump to : navigation , search This article contains wording that promotes the
subject in a subjective manner without imparting real information . Please remove or replace such wording and instead of making proclamations about a subject 's
importance , use facts and attribution to demonstrate that importance . ( August 2017 ) ( Learn how and when to remove this template message ) Prince Edward
Island The coast of Prince Edward Island around Cavendish Geography Location Gulf of Saint Lawrence Area 5,683.91 km ( 2,194.57 sq mi ) Highest elevation 152 m
( 499 ft ) Administration Canada Province Prince Edward Island Largest settlement Charlottetown ( pop . 32,174 ) Demographics Population 139,407 ( 2008 ) Pop .
</item 1>

<item 2>
Geography of Prince Edward Island
( See also a list of communities in Prince Edward Island . ) The island 's landscape is pastoral : rolling hills , pristine forests , white sand beaches , ocean
coves and the famous red soil have given Prince Edward Island a reputation as a province of outstanding natural beauty . A number of laws have been passed by
the provincial government to attempt to preserve the landscape through regulation , although the lack of consistent enforcement and absence of province - wide
zoning and land - use planning has resulted in some aesthetically displeasing development in recent years . The island 's lush landscape has had a strong
bearing not only on its economy but also its culture . Author Lucy Maud Montgomery drew inspiration from the land during the late Victorian Era for the setting
of her classic novel Anne of Green Gables . Today , many of the same qualities that Montgomery and others found in the Island are enjoyed by tourists who visit
during all seasons . They enjoy a variety of leisure activities , including beaches , various golf courses , eco-tourism adventures , and simply touring the
countryside and enjoying cultural events in local communities around the island .
</item 2>

<item 3>
Geography of New Caledonia
Click to enlarge . New Caledonia is made up of a main island , the Grande Terre , and several smaller islands , the Belep archipelago to the north of the Grande
Terre , the Loyalty Islands to the east of the Grande Terre , the ( Isle of Pines ) to the south of the Grande Terre , the Chesterfield Islands and Bellona
Reefs further to the west . Each of these four island groups has a different geological origin : The New Caledonia archipelago , which includes Grande Terre ,
Belep , and the Île des Pins was born as a series of folds of the earth 's mantle between the Permian period ( 251 -- 299 mya ) and the Paleogene and Neogene
periods ( 1.5 -- 66 mya ) . This mantle obduction created large areas of peridotite and a bedrock rich in nickel . The Loyalty Islands , a hundred kilometers to
the east , are coral and limestone islands built on top of ancient collapsed volcanoes originating due to subduction at the Vanuatu trench .
</item 3>

<item 4>
On the Origin of Species
The coastal sea life of the Atlantic and Pacific sides of Central America had almost no species in common even though the Isthmus of Panama was only a few miles
wide . His explanation was a combination of migration and descent with modification . He went on to say : `` On this principle of inheritance with modification
, we can understand how it is that sections of genera , whole genera , and even families are confined to the same areas , as is so commonly and notoriously the
case . '' Darwin explained how a volcanic island formed a few hundred miles from a continent might be colonised by a few species from that continent . These
species would become modified over time , but would still be related to species found on the continent , and Darwin observed that this was a common pattern .
Darwin discussed ways that species could be dispersed across oceans to colonise islands , many of which he had investigated experimentally . Chapter XII
continues the discussion of biogeography . After a brief discussion of freshwater species , it returns to oceanic islands and their peculiarities ; for example
on some islands roles played by mammals on continents were played by other animals such as flightless birds or reptiles .
</item 4>

<item 5>
Geography of Prince Edward Island
32,174 ) Demographics Population 139,407 ( 2008 ) Pop . density 24.53 / km ( 63.53 / sq mi ) Ethnic groups Scottish ( 38.0 % ) , English ( 28.7 % ) , Irish (
27.9 % ) , French ( 21.3 % ) , German ( 4.0 % ) , and Dutch ( 3.1 % ) Prince Edward Island 's geography is mostly pastoral with red soil , white sand , and
scattered communities . Known as the `` Garden of the Gulf '' , the island is located in the Gulf of Saint Lawrence north of Nova Scotia and east of New
Brunswick , with which it forms the Northumberland Strait . PEI consists of the capital city Charlottetown , as well as urban towns Cornwall and Stratford and a
developing urban fringe . A smaller urban area surrounds Summerside Harbor , situated on the southern shore 40 kilometers west of Charlottetown Harbor , and
consists primarily of the city of Summerside . As with all natural harbors on the island , Charlottetown and Summerside harbors are created by rias .
</item 5>

<item 6>
Geography of Prince Edward Island
The geological properties of a white silica sand found at Basin Head are unique in the province ; the sand grains cause a scrubbing noise as they rub against
each other when walked on . Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours . The magnificent
sand dunes at Greenwich are of particular significance . The shifting , parabolic dune system is home to a variety of birds and rare plants and is also a site
of significant archeological interest .
</item 6>

<item 7>
Geography of Prince Edward Island
See also ( edit ) Geography of Canada portal List of parks in Prince Edward Island List of islands of Prince Edward Island References ( edit ) External links (
edit ) Geography of the provinces and territories of Canada National Geography of Canada Centre of Canada Provinces British Columbia Alberta Saskatchewan
Manitoba Ontario Quebec New Brunswick Nova Scotia Prince Edward Island Newfoundland and Labrador Territories Yukon Northwest Territories Nunavut See also
Regions of Canada Islands Rivers Lakes Mountains National Parks Volcanoes Portal WikiProject Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Geography_of_Prince_Edward_Island&oldid=835998214 '' Categories : Geography of Prince Edward Island Hidden categories
: Articles with peacock terms from August 2017 All articles with peacock terms Talk Contents About Wikipedia Français Edit links This page was last edited on 12
April 2018 , at 01 : 54 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you
agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About
Wikipedia
</item 7>

<item 8>
Ring of Fire
Peninsula Quebec City -- Windsor Corridor Peace River Country Cypress Hills Palliser 's Triangle Canadian Shield Interior Alaska - Yukon lowland taiga
Newfoundland ( island ) Vancouver Island Gulf Islands Strait of Georgia Canadian Arctic Archipelago Labrador Peninsula Gaspé Peninsula Avalon Peninsula Bay de
Verde Peninsula Brodeur Peninsula Melville Peninsula Bruce Peninsula Banks Peninsula ( Nunavut ) Cook Peninsula Gulf of Boothia Georgian Bay Hudson Bay James
Bay Greenland United States Pacific Northwest Inland Northwest Northeast New England Mid-Atlantic Commonwealth West Midwest Upper Midwest Mountain States
Intermountain West Basin and Range Province Oregon Trail Mormon Corridor Calumet Region Southwest Old Southwest Llano Estacado Central United States Tallgrass
prairie South South Central Deep South Upland South Four Corners East Coast West Coast Gulf Coast Third Coast Coastal states Eastern United States Appalachia
Trans - Mississippi Great North Woods Great Plains Interior Plains Great Lakes Great Basin Great Basin Desert Acadia Ozarks Ark - La - Tex Waxhaws Siouxland
Twin Tiers Driftless Area Palouse Piedmont Atlantic coastal plain Outer Lands Black Dirt Region Blackstone Valley Piney Woods Rocky Mountains Mojave Desert The
Dakotas The Carolinas Shawnee Hills San Fernando Valley Tornado Alley North Coast Lost Coast Emerald Triangle San Francisco Bay
</item 8>

<item 9>
Dead Sea
Northwest British Isles English Channel Channel Islands Cotentin Peninsula Normandy Brittany Gulf of Lion Iberia Al - Andalus Baetic System Pyrenees Alpide belt
South Italian Peninsula Insular Italy Tuscan Archipelago Aegadian Islands Iberia Al - Andalus Baetic System Gibraltar Arc Southeastern Mediterranean Crimea
Alpide belt Germanic Celtic Slavic countries Uralic European Plain Eurasian Steppe Pontic -- Caspian steppe Wild Fields Pannonian Basin Great Hungarian Plain
Little Hungarian Plain Eastern Slovak Lowland Regions of North America Northern Eastern Canada Western Canada Canadian Prairies Central Canada Northern Canada
Atlantic Canada The Maritimes French Canada English Canada Acadia Acadian Peninsula Quebec City -- Windsor Corridor Peace River Country Cypress Hills Palliser
's Triangle Canadian Shield Interior Alaska - Yukon lowland taiga Newfoundland ( island ) Vancouver Island Gulf Islands Strait of Georgia Canadian Arctic
Archipelago Labrador Peninsula Gaspé Peninsula Avalon Peninsula Bay de Verde Peninsula Brodeur Peninsula Melville Peninsula Bruce Peninsula Banks Peninsula (
Nunavut ) Cook Peninsula Gulf of Boothia Georgian Bay Hudson Bay James Bay Greenland Pacific Northwest Inland Northwest Northeast New England Mid-Atlantic
Commonwealth West Midwest Upper Midwest Mountain States Intermountain West Basin and Range Province Oregon Trail
</item 9>

<item 10>
Geography of Prince Edward Island
The highest point of land is located at Springton in Queens County ( 152 metres or 499 feet above sea level ) . Contents ( hide ) 1 Communities 2 Coastline 3
See also 4 References 5 External links Communities ( edit ) The smaller rural communities as well as the towns and villages throughout the province proudly
retain a slower - paced , old world flavor , something that factors heavily into Prince Edward Island 's popularity as a destination for relaxation . Most rural
communities on Prince Edward Island are based on small - scale agriculture , given that the size of farm properties is quite small when compared with other
areas in Canada . There is an increasing amount of industrial farming as older farm properties are consolidated and modernized . Coastline ( edit ) The
coastline of the island consists of a combination of long beaches , dunes , red sandstone cliffs , salt water marshes and numerous bays and harbours . The
beaches , dunes and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration which oxidizes upon exposure to the air .
</item 10>
=== END ITEM id=7f13f56cf1ac55e4 ===

=== ITEM id=bf327cdb6a2ab901 ===
Gold sentences and facts:
Sentence 0: The Nightmare Before Christmas originated in a poem written by Tim Burton in 1982 , while he was working as an animator at Walt Disney Feature Animation . With the success of Vincent in the same year , Walt Disney Studios began to consider developing The Nightmare Before Christmas as either a short film or 30 - minute television special . Over the years , Burton 's thoughts regularly returned to the project , and in 1990 , he made a development deal with Disney . Production started in July 1991 in San Francisco ; Disney released the film through its Touchstone Pictures banner because the studio believed the film would be `` too dark and scary for kids '' .
  Fact 0.0: The Nightmare Before Christmas originated in a poem written by Tim Burton in 1982.
  Fact 0.1: Tim Burton wrote the poem in 1982 while he was working as an animator at Walt Disney Feature Animation.
  Fact 0.2: Walt Disney Studios began to consider developing The Nightmare Before Christmas in the same year as the success of Vincent.
  Fact 0.3: Walt Disney Studios considered developing The Nightmare Before Christmas as either a short film or a 30-minute television special.
  Fact 0.4: Tim Burton's thoughts regularly returned to The Nightmare Before Christmas over the years.
  Fact 0.5: Tim Burton made a development deal with Disney in 1990.
  Fact 0.6: Production of The Nightmare Before Christmas started in July 1991 in San Francisco.
  Fact 0.7: Disney released The Nightmare Before Christmas through its Touchstone Pictures banner.
  Fact 0.8: Disney released the film through Touchstone because the studio believed the film would be 'too dark and scary for kids'.

Retrieved items:
<item 1>
*The Nightmare Before Christmas* follows Jack Skellington of Halloween Town, who discovers Christmas Town and attempts to celebrate Christmas with comic,
sinister results. Tim Burton conceived it as a poem in 1982; production began in 1991, with Henry Selick directing while Burton was busy with *Batman Returns*.
Danny Elfman composed the songs and score and sang Jack; the voice cast includes Chris Sarandon, Catherine O’Hara, and Ken Page. Released by Disney’s Touchstone
label
</item 1>

<item 2>
The Nightmare Before Christmas
He created concept art and storyboards for the project in collaboration with Rick Heinrichs , who also sculpted character models ; Burton later showed his and
Heinrichs ' works - in - progress to Henry Selick , also a Disney animator at the time . After the success of Vincent in 1982 , Disney started to consider
developing The Nightmare Before Christmas as either a short film or 30 - minute holiday television special . However , the project 's development eventually
stalled , as its tone seemed `` too weird '' to the company . As Disney was unable to `` offer his nocturnal loners enough scope '' , Burton was fired from the
studio in 1984 , and went on to direct the commercially successful films Beetlejuice and Batman . Director Henry Selick ( left ) and producer Tim Burton ( right
) on the Nightmare Before Christmas set . Over the years , Burton regularly thought about the project . In 1990 , Burton found out that Disney still owned the
film rights . He and Selick committed to produce a full - length film with the latter as director . Disney was looking forward to Nightmare `` to show
capabilities of technical and storytelling achievements that were present in Who Framed Roger Rabbit . ''
</item 2>

<item 3>
*The Nightmare Before Christmas* was a critical and financial success, earning over $76 million in its initial run and becoming the first animated film
nominated for the Academy Award for Best Visual Effects. Disney reissued it annually in Disney Digital 3-D from 2006 to 2009, making it the first stop-motion
feature fully converted to 3-D. Its fantasy setting, Halloween Town, is populated by monsters, ghosts, ghouls, goblins, zombies, vampires, werewolves
</item 3>

<item 4>
The Nightmare Before Christmas
The Nightmare Before Christmas - wikipedia The Nightmare Before Christmas Jump to : navigation , search This article is about the film . For other uses , see
The Nightmare Before Christmas ( disambiguation ) . The Nightmare Before Christmas Theatrical release poster Directed by Henry Selick Produced by Tim Burton
Denise Di Novi Screenplay by Caroline Thompson Story by Michael McDowell ( adaptation ) Based on Story and characters by Tim Burton Starring Chris Sarandon
Danny Elfman Catherine O'Hara William Hickey Glenn Shadix Paul Reubens Ken Page Ed Ivory Music by Danny Elfman Cinematography Pete Kozachik Edited by Stan Webb
Production companies Touchstone Pictures Skellington Productions Distributed by Buena Vista Pictures Release date October 29 , 1993 ( 1993 - 10 - 29 ) Running
time 76 minutes Country United States Language English Budget $18 million Box office $76.2 million The Nightmare Before Christmas ( also known as Tim Burton 's
The Nightmare Before Christmas ) is a 1993 American stop - motion animated musical dark fantasy film directed by Henry Selick , and produced and conceived by
Tim Burton .
</item 4>

<item 5>
The Nightmare Before Christmas
The film was met with both critical and financial success , grossing over $76 million during its initial run . It was nominated for the Academy Award for Best
Visual Effects , a first for an animated film . The film has since been reissued by Walt Disney Pictures , and was re-released annually in Disney Digital 3 - D
from 2006 until 2009 , making it the first stop - motion animated feature to be entirely converted to 3D . Contents ( hide ) 1 Plot 2 Voice cast 3 Production
3.1 Development 3.2 Filming 4 Soundtracks 5 Release 5.1 Home media 5.2 Marketing 6 Reception 6.1 Box office 6.2 Critical response 6.3 Accolades 7 Possible
sequel 8 Related media 9 See also 10 Notes 11 References 12 Further reading 13 External links Plot ( edit ) Halloween Town is a fantasy world filled with
citizens such as deformed monsters , ghosts , ghouls , goblins , zombies , mummies , vampires , werewolves , and witches .
</item 5>

<item 6>
*The Nightmare Before Christmas* is a 1993 American stop-motion animated musical dark fantasy film, directed by Henry Selick and conceived and produced by Tim
Burton. It follows Halloween Town’s Jack Skellington as he discovers Christmas and attempts to bring it to his world. Featuring music by Danny Elfman and voices
including Chris Sarandon and Catherine O’Hara, the 76-minute film was released October 29, 1993. Its $18 million budget yielded $76.2 million
</item 6>

<item 7>
Tim Burton and Rick Heinrichs developed concept art and storyboards for *The Nightmare Before Christmas*. Disney initially considered a short or TV special, but
shelved it as too strange and fired Burton in 1984. In 1990, Burton and director Henry Selick revived it as a feature film. It earned $50 million in its initial
U.S. run and $8.7 million in a 2006 reissue. Burton resisted sequels to preserve the film’s integrity, though
</item 7>

<item 8>
The Nightmare Before Christmas
It tells the story of Jack Skellington , a resident from `` Halloween Town '' who stumbles through a portal to `` Christmas Town '' and decides to celebrate the
holiday , with some dastardly and comical consequences . Danny Elfman wrote the songs and score , and provided the singing voice of Jack . The principal voice
cast also includes Chris Sarandon , Catherine O'Hara , William Hickey , Ken Page , Paul Reubens , and Glenn Shadix . The Nightmare Before Christmas originated
in a poem written by Tim Burton in 1982 , while he was working as an animator at Walt Disney Feature Animation . With the success of Vincent in the same year ,
Walt Disney Studios began to consider developing The Nightmare Before Christmas as either a short film or 30 - minute television special . Over the years ,
Burton 's thoughts regularly returned to the project , and in 1990 , he made a development deal with Disney . Production started in July 1991 in San Francisco ;
Disney released the film through its Touchstone Pictures banner because the studio believed the film would be `` too dark and scary for kids '' .
</item 8>

<item 9>
The Nightmare Before Christmas
Other songs included are original holiday songs changed to incorporate the theme of the movie . The last song on the list , however , is the soundtrack for the
Disneyland Haunted Mansion Holiday ride . Release ( edit ) Disney decided to release the film under their Touchstone Pictures banner because they thought the
film would be `` too dark and scary for kids '' , Selick remembered . `` Their biggest fear , and why it was kind of a stepchild project , ( was ) they were
afraid of their core audience hating the film and not coming . '' To convey Burton 's involvement and attract a wider audience , Disney marketed the film as Tim
Burton 's The Nightmare Before Christmas . Burton explained that , `` ... it turned more into more of a brand - name thing , it turned into something else ,
which I 'm not quite sure about . '' The film premiered at the New York Film Festival on October 9 , and was given a limited release on October 15 , 1993 ,
before its wide theatrical release on October 29 .
</item 9>

<item 10>
*The Nightmare Before Christmas* (1993) is a stop-motion musical dark fantasy conceived and produced by Tim Burton and directed by Henry Selick. It follows
Halloween Town’s Jack Skellington (voiced by Chris Sarandon, sung by Danny Elfman), who discovers Christmas and disastrously tries to take it over before making
amends. Catherine O’Hara voices Sally and Shock; Ken Page voices Oogie Boogie. Made with 227 puppets, the film earned about $76
</item 10>
=== END ITEM id=bf327cdb6a2ab901 ===

=== ITEM id=6b812ceff515faf4 ===
Gold sentences and facts:
Sentence 0: Taxpayers in the United States may face various penalties for failures related to Federal , state , and local tax matters . The Internal Revenue Service ( IRS ) is primarily responsible for charging these penalties at the Federal level . The IRS can assert only those penalties specified imposed under Federal tax law . State and local rules vary widely , are administered by state and local authorities , and are not discussed herein .
  Fact 0.0: Taxpayers in the United States may face various penalties for failures related to federal tax matters
  Fact 0.1: Taxpayers in the United States may face various penalties for failures related to state tax matters
  Fact 0.2: Taxpayers in the United States may face various penalties for failures related to local tax matters
  Fact 0.3: The Internal Revenue Service (IRS) is primarily responsible for charging these penalties at the federal level
  Fact 0.4: The IRS can assert only those penalties specified under federal tax law
  Fact 0.5: State and local tax penalty rules vary widely
  Fact 0.6: State and local tax penalty rules are administered by state and local authorities
  Fact 0.7: State and local tax penalty rules are not discussed in the source text

Retrieved items:
<item 1>
IRS penalties
Willis , Hoffman , et al , South - Western Federal Taxation , 2009 Edition ISBN 978 - 0 - 324 - 66052 - 4 , pages 26 -- 13 through 26 - 19 . References ( edit )
Jump up ^ Most of the penalty rules are in Chapters 68 through 75 of the Internal Revenue Code , sections 6651 through 7344 . 26 USC Subtitle F . Jump up ^ Alan
J. Tarr , J.D. , LL. M. ( Taxation ) & Pamela Jensen Drucker , J.D. , LL. M. ( Taxation ) , Civil Tax Penalties , U.S. Income Portfolios , Vol. 634 ( 2d ed.
2012 ) , Bloomberg BNA . Jump up ^ 26 USC 6654 . Jump up ^ 26 USC 6651 ( a ) ( 3 ) . Jump up ^ See IRS Tax Topic 653 . Jump up ^ 26 USC 6651 ( a ) ( 1 ) .
</item 1>

<item 2>
IRS penalties
This penalty of 20 % or 40 % of the increase in tax is due in the case of substantial understatement of tax , substantial valuation misstatements , transfer
pricing adjustments , or negligence or disregard of rules or regulations . For example , a valuation overstatement can result in a 30 % penalty on the amount of
tax owed . Special rules apply for each of these types of errors under which the penalty may be waived . Penalties in connection with information returns ( edit
) Certain types of returns , called `` information returns , '' do not require payment of tax . These include forms filed by employers to report wages ( Form W
- 2 ) and businesses to report certain payments ( Form 1099 series instructions ) . The penalty for failures related to these forms is a dollar amount per form
not timely filed , and the amount of penalty increases with the degree of lateness . The current maximum penalty for these forms is $50 . Many of the forms must
be filed electronically , and filing on paper is considered non-filing . Late filing of returns of partnership income ( Form 1065 ) can result in penalties of
$195 per month per partner , up to a maximum of 12 months .
</item 2>

<item 3>
IRS penalties
Similar penalties may apply to an income tax return ( Form 1120S ) for an S corporation return . 100 % penalty on unpaid withholding taxes ( edit ) Employers
are required to withhold income and social security taxes from wages paid to employees , and to pay these amounts promptly to the government . A penalty of 100
% of the amount not paid over ( plus liability for paying the withheld amounts ) may be collected without judicial proceedings from each and every person who
had custody and control of the funds and did not make the payment to the government . This applies to company employees and officers as individuals , as well as
to companies themselves . There have been reported cases of the IRS seizing houses of those failing to pay over employee taxes . Penalties for failure to
provide foreign information ( edit ) Taxpayers who are shareholders of controlled foreign corporations must file Form 5471 with respect to each such controlled
foreign corporation . Penalties for failure to timely file are $10,000 to $50,000 per form , plus possible loss of foreign tax credits . U.S.
</item 3>

<item 4>
IRS penalties
Tax adviser penalties ( edit ) Penalties also apply to people who promote tax shelters or who fail to maintain and disclose lists of reportable transactions
their customers or clients for those transactions . These monetary penalties can be severe . Judicial appeal of penalties ( edit ) Most penalties are subject to
judicial review . However , the courts rarely modify assessment of the penalties and interest for underestimate or late payment . Further reading ( edit )
Meldman , Robert E. , Federal Taxation Practice and Procedure , ISBN 978 - 0 - 8080 - 1647 - 2 , Chapters 9 and 18 Shafiroff , Ira L. , Internal Revenue Service
Practice and Procedure Deskbook , Kindle edition ( no ISBN ) , or hardcover ISBN 978 - 0 - 87224 - 109 - 1 Pratt , Kulsrud , et al , Federal Taxation , 2006
Edition ISBN 978 - 0 - 7593 - 5175 - 2 , pages 2 -- 6 through 2 - 15 .
</item 4>

<item 5>
IRS penalties
Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines . Contents ( hide ) 1 Underestimate and late payment
penalties 2 Penalties for failure to timely file returns or timely pay tax 3 Accuracy related penalties 4 Penalties in connection with information returns 5 100
% penalty on unpaid withholding taxes 6 Penalties for failure to provide foreign information 7 Excise taxes as penalties 8 Tax fraud penalties 9 Tax adviser
penalties 10 Judicial appeal of penalties 11 Further reading 12 References Underestimate and late payment penalties ( edit ) Taxpayers are required to have
withholding of tax or make quarterly estimated tax payments before the end of the tax year . Because accurate estimation requires accurate prediction of the
future , taxpayers sometimes underestimate the amount due . The penalty for paying too little estimated tax or having too little tax withheld is computed with
interest on the amount that should have been , but was not , paid . Where a taxpayer has filed an income or excise tax return that shows a balance due but does
not pay that balance by the due date of the return ( without extensions ) , a different charge applies .
</item 5>

<item 6>
IRS penalties
Penalty for Failure to Timely Pay Tax : If a taxpayer fails to pay the balance due shown on the tax return by the due date ( even if the reason of nonpayment is
a bounced check ) , there is a penalty of 0.5 % of the amount of unpaid tax per month ( or partial month ) , up to a maximum of 25 % . Penalty for Failure to
Timely Pay After Issuance of Notice : If a taxpayer fails to pay any additional tax assessed by the IRS ( usually as a result of an audit ) the taxpayer may be
liable for a penalty equal to 0.5 % for each month ( or partial month ) during which the failure continues , if the amount is not paid within 21 calendar days
after the date of an IRS notice demanding the payment . If both the failure to file and the failure to pay penalties apply during the same month , then the
failure to file penalty is reduced by 0.5 % each month . The 25 % cap above applies to the 5 % late filing penalty and the 0.5 % late payment penalty together .
The late filing penalty may be waived or abated on showing of reasonable cause for failure .
</item 6>

<item 7>
IRS penalties
Some of these taxes require purchase of tax stamps or other evidence of advance payment of tax . Some require collection of the tax by retailers . An assortment
of penalties apply to manufacturers and retailers not complying with the particular rules . In addition , certain penalties are imposed in the form of an excise
tax . Pension and benefit plans must pay a tax for a variety of failures . Charities and private foundations must pay an excise tax on prohibited transactions
and other failures . Tax fraud penalties ( edit ) Intentional filing of materially false tax returns is a criminal offence . A person convicted of committing
tax fraud , or aiding and abetting another in committing tax fraud , may be subject to forfeiture of property and / or jail time . Conviction and sentencing is
through the court system . Responsibility for prosecution falls to the U.S. Department of Justice , not the Internal Revenue Service . Penalties may be assessed
against tax protesters who raise arguments that income tax laws are not valid , or who otherwise file frivolous returns or court petitions .
</item 7>

<item 8>
IRS penalties
IRS penalties - wikipedia IRS penalties Jump to : navigation , search This article is part of a series on Taxation in the United States of America Federal
taxation ( show ) Alternative minimum tax Capital gains tax Corporate tax Estate tax Excise tax Gift tax Income tax Payroll tax Internal Revenue Service ( IRS )
Internal Revenue Code ( IRC ) IRS tax forms Revenue by state History Constitutional authority Taxpayer standing Court Protest Evasion State and local taxation (
show ) State income tax Property tax Sales tax Use tax Land value tax State tax levels Federal tax reform ( show ) Automated payment transaction tax 9 -- 9 -- 9
Competitive Tax Plan Efficient Taxation of Income FairTax Flat tax Hall -- Rabushka flat tax Kemp Commission Taxpayer Choice Act USA Tax Value added tax Border
- adjustment tax United States portal Taxpayers in the United States may face various penalties for failures related to Federal , state , and local tax matters
. The Internal Revenue Service ( IRS ) is primarily responsible for charging these penalties at the Federal level . The IRS can assert only those penalties
specified imposed under Federal tax law . State and local rules vary widely , are administered by state and local authorities , and are not discussed herein .
</item 8>

<item 9>
IRS penalties
Clair , 45 Amer . Fed . Tax Rep. 2d 80 - 1528 , 80 - 1 U.S. Tax Cas . ( CCH ) ¶ 9364 , ( N.D. Texas 1980 ) , in which the court upheld seizure of the taxpayer
's personal residence for failure to pay over withholding taxes payable by the company . Jump up ^ 26 USC 6038 . Jump up ^ 26 USC 6038A . Jump up ^ 26 USC 6689
. Jump up ^ 26 USC 6677 . Jump up ^ 26 USC 6038B . Jump up ^ Subtitle D of 26 USC . Jump up ^ Manufacturers Taxes , IRS Jump up ^ 26 USC 4971 - 4980G Jump up ^
26 USC 4911 , 4912 and 26 USC 4940 - 4948 . Jump up ^ 26 USC 7301 - 7328 . Jump up ^ 26 USC 7201 - 7217 . Jump up ^ 26 USC 6702 26 USC 6673 .
</item 9>

<item 10>
IRS penalties
Penalties may be monetary or may involve forfeiture of property . Criminal penalties may include jail time , but are imposed only by a federal judge after a
defendant is convicted . Most monetary penalties are based on the amount of tax not properly paid . Penalties may increase with the period of nonpayment . Some
penalties are fixed dollar amounts or fixed percentages of some measure required to be reported . Excise taxes used as penalties are imposed in the Code
sections relating to particular kinds of transactions . Some penalties may be waived or abated where the taxpayer shows reasonable cause for the failure .
Penalties apply for failures to file income tax returns or information returns , or for filing incorrect returns . Some penalties may be very minor . Penalties
apply for certain types of errors on tax returns , and may be substantial . Some penalties are imposed as excise taxes on particular transactions . Certain
other penalties apply for other types of failures . Certain acts may result in forfeiture of property of the taxpayer . There are over 150 kinds of civil
penalties in the U.S. Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines .
</item 10>
=== END ITEM id=6b812ceff515faf4 ===
