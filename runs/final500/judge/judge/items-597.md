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

This file has 12 items.

=== ITEM id=3e0df0d55917d53f ===
Gold sentences and facts:
Sentence 0:  "Pinus" is the sole genus in the subfamily Pinoideae.
  Fact 0.0: Pinus is the sole genus in the subfamily Pinoideae.
Sentence 1:  The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35 unresolved species and many more synonyms.
  Fact 1.0: The Plant List was compiled by the Royal Botanic Gardens, Kew and the Missouri Botanical Garden.
  Fact 1.1: The Plant List accepts 126 species names of pines as current.
  Fact 1.2: The Plant List includes 35 unresolved species of pines.
  Fact 1.3: The Plant List includes many more synonyms.
Sentence 2:  It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
  Fact 2.0: The genus is sometimes considered to have only two species, B. monosperma and B. superba.
  Fact 2.1: The genus is sometimes expanded to include four or five species.

Retrieved items:
<item 1>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 1>

<item 2>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 2>

<item 3>
Thonningia
Thonningia is a monotypic genus of flowering plants in the family Balanophoraceae containing the single species Thonningia sanguinea.

 It is distributed throughout much of southern and western Africa, particularly the tropical regions.

 Common names for the plant include ground pineapple.

 A familiar plant to humans, it has an extremely long list of common names in many African languages.

 Many names are inspired by the resemblance of the plant's inflorescence to a pineapple or palm tree.

 Some of the names can be translated as pineapple of the bush (from Anyi), duiker's kolanut (from Igala), and crown of the ground (from Yoruba).
</item 3>

<item 4>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 4>

<item 5>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 5>

<item 6>
Pine
A pine is any conifer in the genus Pinus, , of the family Pinaceae.

 "Pinus" is the sole genus in the subfamily Pinoideae.

 The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35
unresolved species and many more synonyms.
</item 6>

<item 7>
Broughtonia
Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.

 The genus is abbreviated Bro in trade journals.
</item 7>

<item 8>
Oak
An oak is a tree or shrub in the genus Quercus ( ; Latin "oak tree") of the beech family, Fagaceae.

 There are approximately 600 extant species of oaks.

 The common name "oak" also appears in the names of species in related genera, notably "Lithocarpus" (stone oaks), as well as in those of unrelated species such
as "Grevillea robusta" (silky oaks) and the Casuarinaceae (she-oaks).

 The genus "Quercus" is native to the Northern Hemisphere, and includes deciduous and evergreen species extending from cool temperate to tropical latitudes in
the Americas, Asia, Europe, and North Africa.

 North America contains the largest number of oak species, with approximately 90 occurring in the United States, while Mexico has 160 species of which 109 are
endemic.

 The second greatest center of oak diversity is China, which contains approximately 100 species.
</item 8>

<item 9>
Yellow pine
In ecology and forestry, yellow pine refers to a number of pine species which tend to grow in similar plant communities and yield similar strong wood.

 In the Western United States, yellow pine refers to Jeffrey pine or ponderosa pine.

 In the Southern United States, yellow pine refers to longleaf pine, shortleaf pine, slash pine, or Loblolly pine.

 In the United Kingdom, yellow pine refers to Eastern white pine or Scots pine.
</item 9>

<item 10>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 10>
=== END ITEM id=3e0df0d55917d53f ===

=== ITEM id=4b3277bb1a2d9b18 ===
Gold sentences and facts:
Sentence 0: The University of California, Santa Barbara (commonly referred to as UC Santa Barbara or UCSB) is a public research university and one of the 10 campuses of the University of California system.
  Fact 0.0: The University of California, Santa Barbara is commonly referred to as UC Santa Barbara or UCSB
  Fact 0.1: The University of California, Santa Barbara is a public research university
  Fact 0.2: The University of California, Santa Barbara is one of the 10 campuses of the University of California system
Sentence 1:  Tracing its roots back to 1891 as an independent teachers' college, UCSB joined the University of California system in 1944 and is the third-oldest general-education campus in the system.
  Fact 1.0: UCSB traces its roots back to 1891
  Fact 1.1: UCSB's roots as an independent teachers' college date to 1891
  Fact 1.2: UCSB joined the University of California system in 1944
  Fact 1.3: UCSB is the third-oldest general-education campus in the University of California system
Sentence 2:  It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.
  Fact 2.0: UCSB was founded in 1991
  Fact 2.1: UCSB was founded by the renowned philanthropist Hakim Said
  Fact 2.2: Hakim Said is a philanthropist of the Hamdard Foundation

Retrieved items:
<item 1>
Santa Barbara International Marathon
The Santa Barbara International Marathon and Half Marathon is an annual race that takes place in Goleta, California and Santa Barbara, California.

 The inaugural race was on December 6, 2009.

 For the last five years the Santa Barbara City College has been the official host for the finish line and general operations.

 Rusty and June Snow are the co-directors of the race with Dan Campbell as the technical director.

 The Snow's moved from the East Coast and wanted to bring a marathon to Santa Barbara.

 After a few years of getting the race approved and organized it was finally set to happen in 2009, Rusty Snow states, “I learned to be stubborn from running
marathons.”

 The Santa Barbara International Marathon course serves as a Boston Marathon qualifier.
</item 1>

<item 2>
University of Arizona
The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.

 Founded in 1885, the UA was the first university in the Arizona Territory.

 The university operates two medical schools (University of Arizona College of Medicine - Tucson and the University of Arizona College of Medicine - Phoenix)
and is affiliated with the region's only academic medical centers (Banner - University Medical Center Tucson and Banner - University Medical Center Phoenix).

 The university is also home to the James E. Rogers College of Law and numerous other nationally ranked graduate and professional schools.

 During the 2016–2017 academic year, there was a total enrollment of 43,625 students, including 34,072 undergraduates The University of Arizona is governed by
the Arizona Board of Regents.

 The University of Arizona is one of the elected members of the Association of American Universities (an organization of North America's premier research
institutions) and is the only representative from the state of Arizona to this group.
</item 2>

<item 3>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 3>

<item 4>
Santa Barbara City College
Santa Barbara City College (SBCC) is a two-year community college that opened in 1909 and is located on a 74 acre campus in the city of Santa Barbara,
California, USA.

 SBCC was named co-winner of the 2013 Aspen Institute Prize for Community College Excellence, earning it a #1 national ranking.

 The college was recognized for its quality and focus in four areas: facilitating underrepresented and minority student success, student learning outcomes,
degree completion and transfer rates, and labor market success in securing good jobs after college.

 Santa Barbara City College was established by the Santa Barbara High School District in 1909, making it one of the oldest community colleges in California.

 The college was discontinued shortly after World War I, and its work largely taken over by the Santa Barbara State Normal School, which became the Santa
Barbara State College, and later, the University of California, Santa Barbara.
</item 4>

<item 5>
Mission Santa Barbara
Mission Santa Barbara, also known as Santa Barbara Mission, is a Spanish mission founded by the Franciscan order near present-day Santa Barbara, California.

 It was founded by Padre Fermín Lasuén on December 4, 1786, the feast day of Saint Barbara, as the tenth mission for the religious conversion of the indigenous
local Chumash-Barbareño tribe of Native American people.

 The mission is the namesake of the city of Santa Barbara as well as of Santa Barbara County.
</item 5>

<item 6>
Santa Barbara High School
Santa Barbara Senior High School, "Home of the Dons," is situated on a 40 acre campus in Santa Barbara, California and is part of the Santa Barbara Unified
School District.

 One of the oldest high schools in California, it was established in 1875, and moved to its current site in 1924.

 Until the creation of two rival high schools in 1959 and 1966, it was the sole public high school serving the city of Santa Barbara.

 In November 2005, its 18th-century Spanish influenced design was named an official California and City historic landmark.

 Today, Santa Barbara High School has an enrollment of roughly 2,200 pupils and certified staff numbering 108.
</item 6>

<item 7>
University of California, Santa Barbara Library
The University of California, Santa Barbara Library is the university library system of the University of California, Santa Barbara in Santa Barbara,
California.

 The Library includes four facilities: Two libraries (the Main Library and the Music Library) and two annexes (Annex I and Annex II).

 The library has some three million print volumes, 30,000 electronic journals, 34,450 e-books, 900,055 digitized items, five million cartographic items
(including some 467,000 maps and 3.2 million satellite and aerial images), more than 3.7 million pieces of microform, 167,500 sound recordings, and 4,100
manuscripts.

 The Library states that it holds 3.2 mi of manuscript and archival collections.

 According to data collected by the American Library Association, as of July 2010 the UC Santa Barbara Library holds the 99th largest library collection in
America, reporting 2,948,999 volumes.
</item 7>

<item 8>
University of California, Santa Barbara
The University of California, Santa Barbara (commonly referred to as UC Santa Barbara or UCSB) is a public research university and one of the 10 campuses of the
University of California system.

 The main campus is located on a 1022 acre site near Isla Vista, California, United States, 8 mi from Santa Barbara and 100 mi northwest of Los Angeles.

 Tracing its roots back to 1891 as an independent teachers' college, UCSB joined the University of California system in 1944 and is the third-oldest
general-education campus in the system.
</item 8>

<item 9>
Hamdard University
Hamdard University (Urdu: ) is a private research university with campuses in Karachi and Islamabad, Pakistan.

 It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.

 Hamdard is one of the first and the oldest private institutions of higher education in Pakistan.

 In Karachi, Hamdard University is the largest private research university with a campus area of over 350 acres.
</item 9>

<item 10>
Tunku Abdul Rahman University College
Tunku Abdul Rahman University College, abbreviated as TAR UC, is a non-profit, comprehensive, private university college in Malaysia.

 Named after the first Prime Minister of Malaysia, YTM Tunku Abdul Rahman Putra Al-Haj, the institution was founded in 1969 as Tunku Abdul Rahman College or TAR
College then by the Malaysian Chinese Association (MCA).

 In 2013, the college was upgraded to a university college.

 TAR UC is one of the prestigious private university college in Malaysia, renowned for its accounting programme.
</item 10>
=== END ITEM id=4b3277bb1a2d9b18 ===

=== ITEM id=dbfa55a0c38980c1 ===
Gold sentences and facts:
Sentence 0:  The S70 was essentially a facelifted 850 saloon.
  Fact 0.0: The S70 was essentially a facelifted 850 saloon.
Sentence 1: The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.
  Fact 1.0: The Volvo 850 is a compact executive car.
  Fact 1.1: The Volvo 850 was produced by the Swedish manufacturer Volvo Cars.
  Fact 1.2: The Volvo 850 was produced from 1991 to 1996.
Sentence 2:  The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.
  Fact 2.0: The Volvo 850 was the first front-wheel drive vehicle from Volvo to be sold in North America.
  Fact 2.1: The Volvo 850 was the first all-wheel drive Volvo.

Retrieved items:
<item 1>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 1>

<item 2>
Volvo 300 Series
The Volvo 300 Series is a rear wheel drive small family car sold as both a hatchback and (later) a conventional saloon from 1976 to 1991.

 It was launched in the Netherlands shortly after Volvo acquired a major stake in the passenger car division of DAF in 1973.

 The series consisted of the Volvo 340 (previously 343/345) and the later Volvo 360.
</item 2>

<item 3>
Volvo 164
The Volvo 164 is a 4-door, 6-cylinder luxury sedan unveiled by Volvo at the Paris Motor Show early in October 1968 and first sold as a 1969 model.

 46,008 164s were built before the car was superseded by the 264 in 1975.

 The 164 was Volvo's first venture into the luxury segment since the end of PV 60 production in 1950, and was the first six-cylinder Volvo since the PV800 last
produced in 1958.
</item 3>

<item 4>
Volvo P2 platform
The Volvo P2 platform is a global full-size unibody automobile platform developed and produced by Volvo.

 It is designed for single wheelbases and is adaptable to front- or all wheel drive configurations.

 It was developed by the automaker before its 1999 acquisition by Ford Motor Company, and debuted with the 1998 Volvo S80.
</item 4>

<item 5>
Volvo C30
The Volvo C30 is a near-luxury compact hatchback, manufactured and marketed by Volvo Cars for model years 2006-2013, available across its single generation as a
three-door hatchback.

 Powered by inline-four and straight-five engines, the C30 is variant of the Volvo S40/V50/C70 range, sharing the same Ford C1/Volvo P1 platform.

 Volvo marketed the C30 as a premium hatchback / sports coupe.
</item 5>

<item 6>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 6>

<item 7>
Volvo SI6 engine
Volvo designed the SI6 ("short inline 6") straight-6 automobile engine for use in 2007 models.

 An evolution of the company's long-used straight-5 Volvo Modular engine, which itself is an evolution of the Volvo B6304 straight six engine, the SI6 can be
mounted transversely for front wheel drive or all wheel drive applications.

 Despite the added cylinder and displacement, the engine remains compact, and is in fact 1 mm shorter than the previous straight-5.

 The engine was initially offered in two displacements — a 3.0 L turbocharged version and a 3.2 L naturally aspirated version.

 Both offered variable cam timing, though only the turbo version varies both the intake and exhaust valves.

 On top of the variable cam timing used on the intake cam of the naturally aspirated engine it also had variable valve lift by using Cam Profile Switching
(CPS).
</item 7>

<item 8>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 8>

<item 9>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 9>

<item 10>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 10>
=== END ITEM id=dbfa55a0c38980c1 ===

=== ITEM id=fa00f11622072e39 ===
Gold sentences and facts:
Sentence 0: Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a 1760 play by Francesco Ringhieri, "L'Osiride".
  Fact 0.0: Mosè in Egitto (Moses in Egypt) is a three-act opera.
  Fact 0.1: Mosè in Egitto was written by Gioachino Rossini.
  Fact 0.2: Mosè in Egitto has an Italian libretto by Andrea Leone Tottola.
  Fact 0.3: The libretto of Mosè in Egitto was based on a 1760 play by Francesco Ringhieri.
  Fact 0.4: The 1760 play by Francesco Ringhieri is titled L'Osiride.
Sentence 1: Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.
  Fact 1.0: Lucrezia Borgia is a melodramatic opera.
  Fact 1.1: Lucrezia Borgia is in a prologue and two acts.
  Fact 1.2: Lucrezia Borgia is by Gaetano Donizetti.

Retrieved items:
<item 1>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 1>

<item 2>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 2>

<item 3>
The entries span opera, music, theatre, literature and Renaissance history. Bernhard Bötel sang roles from Mozart to Verdi; operas range from Gluck and Hasse to
Rossini and later composers. Commedia dell’arte’s masked stock characters, improvisation and lazzi are described. Musical highlights include Tchaikovsky’s
Second Piano Concerto, Debussy’s *Masques* and Bartók’s quartets. Other topics include Verdi’s Busseto theatre,
</item 3>

<item 4>
Lucrezia Borgia (1940 film)
Lucrezia Borgia is a 1940 Italian historical film directed by Hans Hinrich and starring Isa Pola, Friedrich Benfer and Carlo Ninchi.

 The film portrays the life of Lucrezia Borgia (1480-1519), one of a number of Italian films of the era set during the Renaissance.

 It was made at the Scalera Studios in Rome.
</item 4>

<item 5>
Lucrezia Borgia (1947 film)
Lucrezia Borgia (Spanish:Lucrecia Borgia) is a 1947 Argentine historical film directed by Luis Bayón Herrera and starring Olinda Bozán, Héctor Quintanilla and
Gogó Andreu.

 The film portrays the life of Lucrezia Borgia (1480-1519).
</item 5>

<item 6>
Bernhard Bötel’s roles ranged from Mozart’s Belmonte to Verdi’s Duke of Mantua. The entries survey operas across eras: Gluck’s reform works *Orfeo ed Euridice*
(1762) and *Paride ed Elena* (1770); Hasse’s *Romolo ed Ersilia* (1765); Rossini’s *Mosè in Egitto* (1818); and later works by Donizetti, Bernstein,
</item 6>

<item 7>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 7>

<item 8>
Lucrezia Borgia (play)
Lucrezia Borgia (French: "Lucrèce Borgia" ) is an 1833 play by the French writer Victor Hugo.

 It is a historical work portraying the Renaissance-era Italian aristocrat Lucrezia Borgia.

 The play (along with "Angelo, Tyrant of Padua") is believed to have been a major influence on Oscar Wilde's "The Duchess of Padua" (1891).
</item 8>

<item 9>
Mosè in Egitto
Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a
1760 play by Francesco Ringhieri, "L'Osiride".

 It premièred on 5 March 1818 at the recently reconstructed Teatro San Carlo in Naples, Italy.
</item 9>

<item 10>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 10>
=== END ITEM id=fa00f11622072e39 ===

=== ITEM id=9f21f45ee289889e ===
Gold sentences and facts:
Sentence 0: Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.
  Fact 0.0: Bumble and bumble Products, LLC is one of 27 brands.
  Fact 0.1: The 27 brands are owned by Estée Lauder Companies, Inc.
  Fact 0.2: Bumble and bumble Products, LLC is owned by Estée Lauder Companies, Inc.
Sentence 1:  The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown Manhattan, New York City.
  Fact 1.0: Estée Lauder Companies, Inc. owns a diverse portfolio of brands.
  Fact 1.1: The brand portfolio is distributed internationally through retail channels.
  Fact 1.2: The brand portfolio is distributed internationally through digital commerce.
  Fact 1.3: Estée Lauder Companies, Inc. has its headquarters in Midtown Manhattan, New York City.

Retrieved items:
<item 1>
Universal Pictures
Universal Pictures (also referred to as Universal Studios or simply Universal) is an American film studio owned by Comcast through the Universal Filmed
Entertainment Group division of its wholly owned subsidiary NBCUniversal.

 The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H.
Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film,
and the oldest in terms of the overall film market .

 Its studios are located in Universal City, California, and its corporate offices are located in New York City.
</item 1>

<item 2>
Shaver Transportation Company
The Shaver Transportation Company is an inland water freight transportation company based in Portland, Oregon, United States.

 The company was founded in 1880 and played a major role in the development of freight transport in the Portland area and along the Columbia.
</item 2>

<item 3>
Cornerstone (agency)
Cornerstone is a New York-based creative lifestyle marketing and public relations agency with offices in New York City, Los Angeles, and London.

 It is helmed by co-CEOs Rob Stone and Jon Cohen.

 The company employs approximately 100 professionals worldwide, in addition to a network of field marketing representatives throughout the United States.
</item 3>

<item 4>
Buena Vista (brand)
Buena Vista is a brand name which was historically often used for divisions and subsidiaries of The Walt Disney Company, whose primary studios, the Walt Disney
Studios, are located on Buena Vista Street in Burbank, California.

 (The studio lot is also home to the Company's corporate headquarters, the Team Disney Burbank building.)

 The logos for the various Buena Vista brands featured the "Buena Vista" wordmark superimposed over the Disney Castle image to signal the affiliation between
Buena Vista and Disney.

 The name literally means "good view" in Spanish.
</item 4>

<item 5>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 5>

<item 6>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 6>

<item 7>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 7>

<item 8>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 8>

<item 9>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 9>

<item 10>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 10>
=== END ITEM id=9f21f45ee289889e ===

=== ITEM id=46d228d7f6183c71 ===
Gold sentences and facts:
Sentence 0: Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
  Fact 0.0: Ciaran Gribbin was born in 1976.
  Fact 0.1: Ciaran Gribbin is a Northern Irish singer-songwriter.
  Fact 0.2: Ciaran Gribbin is a guitarist.
  Fact 0.3: Ciaran Gribbin is a producer.
  Fact 0.4: Ciaran Gribbin hails from Castledawson, Northern Ireland.
Sentence 1: Brett Allen Scallions (born December 21, 1971) is an American musician.
  Fact 1.0: Brett Allen Scallions was born on December 21, 1971.
  Fact 1.1: Brett Allen Scallions is an American musician.
Sentence 2:  He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
  Fact 2.0: Brett Allen Scallions is the lead vocalist of the post-grunge band Fuel.
  Fact 2.1: Brett Allen Scallions is the rhythm guitarist of the post-grunge band Fuel.
  Fact 2.2: Brett Allen Scallions is the lyricist of the post-grunge band Fuel.

Retrieved items:
<item 1>
The Best of Fuel
The Best of Fuel is a compilation album by American hard rock band Fuel.

 It contains songs from their first three full-length albums, "Sunburn", "Something Like Human", and "Natural Selection".

 All of the said albums featured vocalist Brett Scallions who departed from the group only a few months after the compilation's release.

 "The Best of Fuel" consists solely of the band's released singles, with the exception of "Million Miles" from "Natural Selection" (it features "Quarter" from
the same album instead).
</item 1>

<item 2>
Soul to Preach To
"Soul to Preach To" is the first single released from Fuel's album "Puppet Strings".

 It is also the first single released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 2>

<item 3>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 3>

<item 4>
Brett Scallions
Brett Allen Scallions (born December 21, 1971) is an American musician.

 He is the lead vocalist, rhythm guitarist, and lyricist of post-grunge band Fuel.
</item 4>

<item 5>
Yeah! (Fuel song)
"Yeah!"

 is the first song released from Fuel's album "Puppet Strings".

 It is also the first song released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 5>

<item 6>
Circus Diablo
Circus Diablo is an American rock band, formed in early 2006 by Billy Morrison (vocals), Billy Duffy (lead guitar) and Ricky Warwick (rhythm guitar).

 Fuel frontman Brett Scallions and Velvet Revolver drummer Matt Sorum subsequently joined the band on bass and drums, respectively.

 To date, Circus Diablo have released one studio album, entitled "Circus Diablo".
</item 6>

<item 7>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 7>

<item 8>
Puppet Strings
Puppet Strings is the fifth studio album by American rock band Fuel.

 Released March 4, 2014, it is their first studio effort since 2003's "Natural Selection" to feature original lead singer Brett Scallions and the first Fuel
album not to feature original songwriter/guitarist, Carl Bell and bassist Jeff Abercrombie.

 Additionally, no one who participated in the recording of Fuel's previous album "Angels & Devils" participated in the recording of "Puppet Strings".
</item 8>

<item 9>
Angels &amp; Devils (Fuel album)
Angels & Devils is the fourth studio album by American rock band Fuel.

 Released on August 7, 2007, it was their first studio effort since 2003's "Natural Selection" and was the last Fuel album to feature original
songwriter/guitarist, Carl Bell and bassist Jeff Abercrombie.

 No musician on this album is still with the band.

 It was also Fuel's only studio album to feature new vocalist Toryn Green, and their final album for Epic Records.

 With a new singer, the album also introduced a new Fuel logo.
</item 9>

<item 10>
Ciaran Gribbin
Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
</item 10>
=== END ITEM id=46d228d7f6183c71 ===

=== ITEM id=69c69028135076da ===
Gold sentences and facts:
Sentence 0: Ionolyce is a small genus of butterflies in the family Lycaenidae.
  Fact 0.0: Ionolyce is a small genus of butterflies.
  Fact 0.1: Ionolyce is in the family Lycaenidae.
Sentence 1:  "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.
  Fact 1.0: I. helicon is widespread in the Indoaustralasian region.
  Fact 1.1: I. helicon is widespread from India to the Malay Archipelago.
Sentence 2: India lies on the Indian Plate, the northern portion of the Indo-Australian Plate, whose continental crust forms the Indian subcontinent.
  Fact 2.0: India lies on the Indian Plate.
  Fact 2.1: The Indian Plate is the northern portion of the Indo-Australian Plate.
  Fact 2.2: The continental crust of the Indian Plate forms the Indian subcontinent.
Sentence 3:  It is the seventh-largest country in the world, with a total area of 3287263 km2 .
  Fact 3.0: India is the seventh-largest country in the world.
  Fact 3.1: India has a total area of 3287263 km2.

Retrieved items:
<item 1>
White-bellied musk deer
The White-bellied musk deer or Himalayan musk deer ("Moschus leucogaster") is a musk deer species occurring in the Himalayas of Nepal, Bhutan, India, Pakistan
and China.

 It is listed as endangered by the IUCN because of overexploitation resulting in a probable serious population decline.
</item 1>

<item 2>
List of companies of India
India is a country in South Asia.

 It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world.
</item 2>

<item 3>
Forests of Australia
Australia has many forests of importance due to significant features, despite being one of the driest continents.

 Australia has approximately 123 million hectares of native forest, which represents about 16% of Australia's land area.

 The majority of Australia's trees are hardwoods, typically eucalypts, rather than softwoods like pine.

 While softwoods dominate some native forests, their total area is judged insufficient to constitute a major forest type in Australia's National Forest
Inventory.

 The Forests Australia website provides up-to-date information on Australia's forests.

 Detailed information on Australia's forests is available from Australia's State of the Forests Reports that are published every five years.
</item 3>

<item 4>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 4>

<item 5>
Marsh deer
The marsh deer ("Blastocerus dichotomus") is the largest deer species from South America reaching a length of 2 m and a shoulder height of 1.2 m .

 It is found in Argentina, Bolivia, Peru, Brazil, Uruguay and Paraguay.

 Formerly found in much of tropical and subtropical South America, it ranged east of the Andes, south from the Amazon rainforest, west of the Brazilian Atlantic
rainforest and north of the Argentinian Pampa.

 Today it is largely reduced to isolated populations at marsh and lagoon zones in the Paraná, Paraguay, Araguaia and Guapore river basins.

 Small populations also occur in the southern Amazon, including Peru where protected in Bahuaja-Sonene National Park.

 It is listed as a vulnerable species by the IUCN and on CITES Appendix I.
</item 5>

<item 6>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 6>

<item 7>
Ionolyce
Ionolyce is a small genus of butterflies in the family Lycaenidae.

 "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.

 "I selkon"and "I.brunnescens" are both endemic to the Solomon Islands.
</item 7>

<item 8>
Common brown leafhopper
The common brown leafhopper, "Orosius orientalis" (Matsumura) (Hemiptera: Cicadellidae) is one of the most common species of Australian leafhoppers with a very
wide host range.

 It is an important vector of several viruses and phytoplasmas worldwide.

 In Australia, phytoplasmas vectored by "O. orientalis" cause a range of economically important diseases including legume little leaf, tomato big bud, lucerne
witches broom, potato purple top wilt, Australian lucerne and is a possible vector of Australian grapevine yellows.

 "O. orientalis" also transmits Tobacco yellow dwarf virus (TYDV genus Mastrevirus, family Geminiviridae) to beans, causing bean summer death disease and to
tobacco, causing tobacco yellow dwarf disease.
</item 8>

<item 9>
India
India, officially the Republic of India ("Bhārat Gaṇarājya"), is a country in South Asia.

 It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world.

 It is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast.

 It shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the northeast; and Myanmar (Burma) and Bangladesh to the east.

 In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives.

 India's Andaman and Nicobar Islands share a maritime border with Thailand and Indonesia.
</item 9>

<item 10>
Consul fabius
Consul fabius, the tiger leafwing, is the most common and well known species of the genus "Consul" of subfamily Charaxinae in the brush-footed butterfly family
(Nymphalidae).

 It is found all over the Neotropics.
</item 10>
=== END ITEM id=69c69028135076da ===

=== ITEM id=8fa136e46b4d35c1 ===
Gold sentences and facts:
Sentence 0: Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.
  Fact 0.0: Bertalan Farkas was born on August 2, 1949.
  Fact 0.1: Bertalan Farkas is the first Hungarian cosmonaut.
  Fact 0.2: Bertalan Farkas is the first Esperantist in space.
Sentence 1: Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
  Fact 1.0: Hans Wilhelm Schlegel was born in Überlingen on 3 August 1951.
  Fact 1.1: Hans Wilhelm Schlegel is a German physicist.
  Fact 1.2: Hans Wilhelm Schlegel is an ESA astronaut.
  Fact 1.3: Hans Wilhelm Schlegel is a veteran of two NASA Space Shuttle missions.

Retrieved items:
<item 1>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 1>

<item 2>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 2>

<item 3>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 3>

<item 4>
Eberhard Köllner
Eberhard Köllner (born 29 September 1939 in Stassfurt, Germany) was selected for Soyuz 31 as the backup for Sigmund Jähn.
</item 4>

<item 5>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 5>

<item 6>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 6>

<item 7>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 7>

<item 8>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 8>

<item 9>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
</item 9>

<item 10>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 10>
=== END ITEM id=8fa136e46b4d35c1 ===

=== ITEM id=c833bcbd33dd30b0 ===
Gold sentences and facts:
Sentence 0:  It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!
  Fact 0.0: The entity referred to as 'It' is known for producing the long-running series 'Tom Goes to the Mayor'.
  Fact 0.1: The entity referred to as 'It' is known for producing the long-running series 'Nathan For You'.
  Fact 0.2: The entity referred to as 'It' is known for producing the long-running series 'Tim and Eric Awesome Show, Great Job!'.
Sentence 1: " and "Check It Out!
  Fact 1.0: 'Check It Out!' is a series title.
Sentence 2:  with Dr. Steve Brule".
  Fact 2.0: The series title is 'Check It Out! with Dr. Steve Brule'.
Sentence 3: Check It Out!
  Fact 3.0: Check It Out! is a title.
Sentence 4:  with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!
  Fact 4.0: Check It Out! with Dr. Steve Brule is an American sketch comedy television series.
  Fact 4.1: Check It Out! with Dr. Steve Brule is a spin-off of 'Tim and Eric Awesome Show, Great Job!'.

Retrieved items:
<item 1>
David Jablin
David Jablin is an award winning comedy producer and director and one of the pioneers of original television programming for Pay Television.

 In 1981 he created and produced the innovative comedy anthology series "Likely Stories" for HBO/Cinemax, which featured the directing debuts of Rob Reiner,
Danny DeVito, Christopher Guest, and Billy Crystal.
</item 1>

<item 2>
Upright Citizens Brigade (TV series)
Upright Citizens Brigade is an American sketch comedy television series that premiered on August 19, 1998 on Comedy Central.

 The show aired for three seasons with each season consisting of ten episodes.

 The series featured four members of Upright Citizens Brigade, an improvisational sketch comedy group.

 The cast included Matt Besser, Amy Poehler, Ian Roberts, and Matt Walsh.

 The cast would later reunite for another series of a similar format that premiered in 2016 on Seeso.
</item 2>

<item 3>
Abso Lutely Productions
Abso Lutely Productions is a film and television production company owned by actor and producers Tim Heidecker, Eric Wareheim and producer Dave Kneebone.

 It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!

" and "Check It Out!

 with Dr. Steve Brule".
</item 3>

<item 4>
Check It Out! with Dr. Steve Brule
Check It Out!

 with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!

" starring John C. Reilly as Dr. Steve Brule.

 The series premiered on Cartoon Network's late night programming block, Adult Swim, on May 16, 2010.

 The program follows Brule as he examines different facets of living.

 His severe naivete and social awkwardness generally land him in embarrassing situations, though he largely remains ignorant of any embarrassment he's causing
himself.

 As the series progresses, he reveals shocking and sometimes horrifying details about his past and personal life.
</item 4>

<item 5>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 5>

<item 6>
In Living Color
In Living Color is an American sketch comedy television series that originally ran on Fox from April 15, 1990, to May 19, 1994.

 Brothers Keenen and Damon Wayans created, wrote and starred in the program.

 The show was produced by Ivory Way Productions in association with 20th Century Fox Television and was taped at stage 7 at the Fox Television Center on Sunset
Boulevard in Hollywood, California.

 The title of the series was inspired by the NBC announcement of broadcasts being presented "in living color" during the 1960s, prior to mainstream color
television.

 It also refers to the fact that most of the show's cast were black, unlike other sketch comedy shows such as "Saturday Night Live" whose casts are mostly
white.

 It was controversial due to the Wayans' decision to portray African-American humor from the ghetto in a time when mainstream American tastes regarding black
comedy had been set by more upscale shows such as "The Cosby Show", causing an eventual feud for control between Fox executives and the Wayans.
</item 6>

<item 7>
Dave Kneebone
Dave Kneebone is an American producer.

 Along with Eric Wareheim and Tim Heidecker he runs Abso Lutely Productions which produces a variety of television and films.

 His role has been described as the business chief and "straight man" at Abso Lutely.

 Kneebone has worked as a producer on a variety of television shows including "Comedy Bang!

 Bang!

", "Nathan for You", and "Tim and Eric Awesome Show, Great Job!

".

 He is also listed a producer on the feature film "Tim and Eric's Billion Dollar Movie".
</item 7>

<item 8>
The Roy Files
The Roy Files is an Irish-based British children's television show, filmed in Dublin, Ireland, which was broadcast by CBBC in the United Kingdom and ABC Me in
Australia.

 It began airing 7 December 2015.

 The show centres on the title character Roy O'Brien, the 11-year-old animated son of a live-action family.

 The series is a spin-off from "ROY", an Irish-British television show which ran from 1 July 2009 to 7 April 2015.

 It stars Jason Cullen (replacing Scott Graham as Roy), Simon Delaney, Cathy Belton and Martha Byrne.
</item 8>

<item 9>
Friends of the People
Friends of the People is an American sketch comedy television series.

 It was slated to premiere on TruTV in summer 2014, but was pushed to October 28, 2014, as part of the network's shift in their programming direction.

 Many of the cast members (Jennifer Bartels, Jermaine Fowler, and Lil Rel Howery) were originally reported to be cast members of a planned revival of In Living
Color which never materialized.

 The show's first season consists of 10 episodes.

 This makes it the network's first sketch comedy show.

 The series holds a TV-14 rating, though select episodes are rated TV-MA--also a first for the truTV network.
</item 9>

<item 10>
Bagboy (TV special)
Bagboy is a 2015 American television special produced for Adult Swim, and aired on February 21, 2015, to positive critical reception.

 Written and directed by Tim Heidecker and Eric Wareheim for Abso Lutely Productions and John C. Reilly, the special is a fictional sitcom pilot in the universe
of "Tim and Eric Awesome Show, Great Job!

", written by and starring Reilly's recurring "Tim and Eric" character Dr. Steve Brule.

 This is the second "Tim and Eric" spin-off to feature the Dr. Steve Brule character after "Check it Out!

 with Dr. Steve Brule".
</item 10>
=== END ITEM id=c833bcbd33dd30b0 ===

=== ITEM id=e8f77c0271836ce3 ===
Gold sentences and facts:
Sentence 0: Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.
  Fact 0.0: Alice Ann Munro's nee name is Laidlaw.
  Fact 0.1: Alice Ann Munro was born on 10 July 1931.
  Fact 0.2: Alice Ann Munro is a Canadian short story writer.
  Fact 0.3: Alice Ann Munro is a Nobel Prize winner.
Sentence 1: Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well as her left-wing sympathies and political activism.
  Fact 1.0: Lillian Florence Hellman was born on June 20, 1905.
  Fact 1.1: Lillian Florence Hellman died on June 30, 1984.
  Fact 1.2: Lillian Florence Hellman was an American dramatist.
  Fact 1.3: Lillian Florence Hellman was an American screenwriter.
  Fact 1.4: Lillian Florence Hellman is known for her success as a playwright on Broadway.
  Fact 1.5: Lillian Florence Hellman is known for her left-wing sympathies.
  Fact 1.6: Lillian Florence Hellman is known for her political activism.

Retrieved items:
<item 1>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 1>

<item 2>
The Little Foxes (film)
The Little Foxes (1941) is an American drama film directed by William Wyler.

 The screenplay by Lillian Hellman is based on her 1939 play of the same name.

 Hellman's ex-husband Arthur Kober, Dorothy Parker and her husband Alan Campbell contributed additional scenes and dialogue.
</item 2>

<item 3>
Rea Wilmshurst
Rea Wilmshurst (August 10, 1941 – March 22, 1996) graduated from the University of Toronto with a degree in English in 1970.

 She went on to edit eight volumes of Lucy Maud Montgomery's previously unknown short stories and publish them through McClelland & Stewart.

 In 1985, she published a bibliography of Montgomery's short stories, poems, and articles.

 Wilmshurst was also an editorial assistant for the projects that compiled the Collected Works of John Stuart Mill and the Collected Works of Samuel Taylor
Coleridge.
</item 3>

<item 4>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 4>

<item 5>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 5>

<item 6>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 6>

<item 7>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 7>

<item 8>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 8>

<item 9>
Alice Walker
Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.

 She wrote the critically acclaimed novel "The Color Purple" (1982) for which she won the National Book Award and the Pulitzer Prize for Fiction.

 She also wrote the novels "Meridian" (1976) and "The Third Life of Grange Copeland" (1970), among other works.
</item 9>

<item 10>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 10>
=== END ITEM id=e8f77c0271836ce3 ===

=== ITEM id=488857bd05337c4d ===
Gold sentences and facts:
Sentence 0: The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
  Fact 0.0: The Consul is an opera.
  Fact 0.1: The Consul is an opera in three acts.
  Fact 0.2: The Consul has music by Gian Carlo Menotti.
  Fact 0.3: The Consul has a libretto by Gian Carlo Menotti.
  Fact 0.4: The Consul is Gian Carlo Menotti's first full-length opera.
Sentence 1:  He completed the music for the opera while living in Zurich in 1916.
  Fact 1.0: Gian Carlo Menotti completed the music for The Consul.
  Fact 1.1: Gian Carlo Menotti completed the music for The Consul in 1916.
  Fact 1.2: Gian Carlo Menotti was living in Zurich in 1916.
  Fact 1.3: Gian Carlo Menotti completed the music for The Consul while living in Zurich.

Retrieved items:
<item 1>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 1>

<item 2>
Niccolò Barbieri
Niccolò (or Nicolò) Barbieri (Vercelli, 1586 - 1641) was an Italian writer and actor of the commedia dell'arte theatrical genre.

 He was also known as Beltrame di Milano ("Beltrame of Milan") in reference to one of his most popular characters, Beltrame; this was the main character of one
of Barbieri's best known plays, "L'inavertito", which is known to have inspired Moliere's "L'Étourdi ou Les Contretemps" ("The Blunderer").

 Besides popularizing Beltrame, Barbieri is also credited with creating another "commedia dell'arte" mask, Scapino.
</item 2>

<item 3>
Arlecchino (opera)
Arlecchino, "oder Die Fenster" ("Harlequin, or The Windows", is a one-act opera with spoken dialog by Ferruccio Busoni, with a libretto in German, composed in
1913.

 He completed the music for the opera while living in Zurich in 1916.

 It is a number opera written in neo-classical style and includes ironic allusions to operatic conventions and situations typical of the late 18th and early
19th centuries.

 It even includes a parody of a duel.
</item 3>

<item 4>
Graph query row:
work: The Consul
type: opera
genre: None
creator: Gian Carlo Menotti
other_work: Arlecchino, oder Die Fenster
other_type: one-act opera
other_genre: None
other_creator: Ferruccio Busoni
creator_connection: None
</item 4>

<item 5>
Pedrolino
Pedrolino is a "primo zanni", or comic servant, of the "Commedia dell'Arte"; the name is a hypocorism of "Pedro" (Peter), via the suffix "-lino".

 The character made its first appearance in the last quarter of the 16th century, apparently as the invention of the actor with whom the role was to be long
identified, Giovanni Pellesini.

 Contemporary illustrations suggest that his white blouse and trousers constituted "a variant of the typical "zanni" suit", and his Bergamasque dialect marked
him as a member of the "low" rustic class.

 But if his costume and social station were without distinction, his dramatic role was certainly not: as a multifaceted "first" "zanni", his character was—and
still is—rich in comic incongruities.
</item 5>

<item 6>
The Consul
The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
</item 6>

<item 7>
Commedia dell'arte
Commedia dell'arte (] , comedy of the profession) was an early form of professional theatre, originating from Italy, that was popular in Europe from the 16th
through the 18th century.

 "Commedia dell'arte" also is known as "commedia alla maschera", "commedia improvviso", and "commedia dell'arte all'improvviso".

 "Commedia" is a form of theatre characterized by masked "types" which began in Italy in the 16th century and was responsible for the advent of actresses
(Isabella Andreini) and improvised performances based on sketches or scenarios.

 A "commedia", such as "The Tooth Puller", is both scripted and improvised.

 Characters' entrances and exits are scripted.

 A special characteristic of "commedia dell'arte" are the "lazzi".

 A "lazzo" is a joke or "something foolish" or "witty".

 Another characteristic of "commedia dell'arte" is pantomime, which is mostly used by the character Arlecchino (Harlequin).
</item 7>

<item 8>
The Carp (opera)
The Carp is a one-act comic opera (styled "a whimsicality") with a libretto by Frank Desprez and music by Alfred Cellier.

 It was first produced at the Savoy Theatre from 13 February 1886 to 19 January 1887, as a companion piece to "The Mikado".

 It was then revived as companion to "Ruddigore" from 21 February 1887 to 5 November 1887.

 The piece also toured throughout 1888.
</item 8>

<item 9>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 9>

<item 10>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
</item 10>
=== END ITEM id=488857bd05337c4d ===

=== ITEM id=717373633cc995cc ===
Gold sentences and facts:
Sentence 0: Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris Institute or Art School, Harris Technical School and the Harris Orphanage.
  Fact 0.0: Edmund Robert Harris lived c. 1804 to 27 May 1877.
  Fact 0.1: Edmund Robert Harris was an English lawyer.
  Fact 0.2: Edmund Robert Harris was from Preston, Lancashire, UK.
  Fact 0.3: Edmund Robert Harris was the principal benefactor of the Harris Museum.
  Fact 0.4: Edmund Robert Harris was the principal benefactor of the Harris Institute or Art School.
  Fact 0.5: Edmund Robert Harris was the principal benefactor of the Harris Technical School.
  Fact 0.6: Edmund Robert Harris was the principal benefactor of the Harris Orphanage.
Sentence 1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
  Fact 1.0: The Harris Museum, Art Gallery & Preston Free Public Library is a museum building in Preston.
  Fact 1.1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed building.

Retrieved items:
<item 1>
Edmund Robert Harris
Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris
Institute or Art School, Harris Technical School and the Harris Orphanage.
</item 1>

<item 2>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 2>

<item 3>
The entries profile notable figures across philanthropy, arts, journalism, religion, academia, business, and politics: Edmund R. Harris funded Preston
institutions; Edmund de Unger built the Keir Islamic-art collection, now loaned to Dallas; Mencken was a major American satirist; and H. Donald Wilson helped
create LexisNexis. Others include leaders, scholars, artists, and advocates such as George Albert Smith, Andrew Benjamin, Inderpal Grewal, Emeka Es
</item 3>

<item 4>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 4>

<item 5>
Edmund de Unger
Edmund Robert Anthony de Unger (Hungarian: "Odon Antal Robert de Unger" , b 6 August 1918, Budapest - d 25 January 2011, Ham, Surrey) was a Hungarian-born
property developer and art collector.

 In London he built up the Keir Collection, one of the greatest post-war collections of Islamic art, bequeathed in 2008 to the Pergamon Museum of Islamic Art in
Berlin.

 The arrangement for the museum to curate the collection came to an end in July 2012.

 The collection is now hosted by the Dallas Museum of Art as of May 2014 for a 15-year renewable loan.
</item 5>

<item 6>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 6>

<item 7>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 7>

<item 8>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 8>

<item 9>
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed building in Preston. Lichfield’s Minster Pool, formed in the 11th century to
power a mill and defend the Cathedral Close, became a public amenity after the mill’s demolition in 1856. Cheshire’s adjacent medieval hunting forests of Mara
and Mondrem covered over 60 square miles and held boar and deer. Dean Mills Reservoir lies high on Winter Hill, north of Bolton.
</item 9>

<item 10>
HSBC Building (Hong Kong)
HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC
Holdings.

 It is located along the southern side of Statue Square near the location of the old City Hall, Hong Kong (built in 1869, demolished in 1933).

 The previous HSBC building was built in 1935 and pulled down to make way for the current building.

 The address remains as 1 Queen's Road Central (the north facing side of the building was served by Des Voeux Road, which was seashore, making Queen's Road was
the main entrance, in contrast to modern day that usually accessed the bank from Des Voeux Road).

 The building can be reached from Exit K of Central MTR Station and facing Statue Square.
</item 10>
=== END ITEM id=717373633cc995cc ===
