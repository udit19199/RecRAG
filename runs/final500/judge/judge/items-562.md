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

This file has 10 items.

=== ITEM id=c53aa5fa87ae57cb ===
Gold sentences and facts:
Sentence 0: PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.
  Fact 0.0: PC Magazine is shortened as PCMag.
  Fact 0.1: PC Magazine is an American computer magazine.
  Fact 0.2: PC Magazine is published by Ziff Davis.
Sentence 1:  The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
  Fact 1.0: The magazine focuses on sculpture.
  Fact 1.1: The magazine focuses on sculpture collectors.
  Fact 1.2: The magazine focuses on sculpture enthusiasts.
  Fact 1.3: Painting is also frequently featured in the magazine.
  Fact 1.4: Poetry is also frequently featured in the magazine.

Retrieved items:
<item 1>
Windows: The Official Magazine
Windows: The Official Magazine is a technology magazine produced by Future plc in association with Microsoft, and published worldwide.

 The title describes itself as "A PC magazine for real life", and contains news, features, guides and reviews designed to show readers how to make better use of
their Windows PC.
</item 1>

<item 2>
Computer Magazine
"Computer Magazine" is a free publication (in addition to their webcasts and other resources) sponsored by the nonprofit UTCP (United Technology and Computing
Professionals) organization, and as such charges no fees for the publication nor is influenced by advertising, so their reviews are relied on in the industry
and considered unbiased and thorough.

 "Computer Magazine" is one of the early large technology publications and resources available on the web still existent and thriving today and that has
remained independent.

 ComputerMagazine.com is a tech news and resources consolidator that publishes part of the site in a semi-time line/blogging format that is popular among their
wide following of subscriber and non-subscriber readers, allowing readers to respond and comment on various articles.

 Site contributors include many of the well known technology authors, experts and publication sources, content and articles are provided by major technology
syndicators and by external expert technology sources (such as "Computer World", "Information Week", "Network World", "Wired," "Time", etc.)
</item 2>

<item 3>
The entries cover popular-science and technology publications, media organizations, and a computer maker. Austrian German-language Universum is financially
supported by the Austrian National Science Foundation; U.S. Popular Science, founded in 1872, is award-winning and internationally distributed. Popular science
explains science for general audiences across media. NASA-sponsored Astrobiology Magazine has reported on space missions and life-universe research since 2000.
Other titles include culture magazine PAPER, PCMag, Windows: The Official Magazine, independent
</item 3>

<item 4>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 4>

<item 5>
Computer Magazine
as well as "Computer Magazine" staff writers, and is currently managed and edited by the industry veteran Christopher Swearingin an MCSE and former CIO and
regarded author as well as contributor/reporter for "Computer Magazine" and other publications.
</item 5>

<item 6>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 6>

<item 7>
Computer Press Association
Founded in 1983, the Computer Press Association (CPA) was established to promote excellence in the field of computer journalism.

 The association was composed of working editors, writers, producers, and freelancers who covered issues related to computers and technology.

 The CPA conducted the annual Computer Press Awards, which was the preeminent editorial awards of the computer and technology media.

 The CPA Awards honored outstanding examples in print, broadcast and electronic media.

 Awards were given for print publications, such as PC Magazine; online news media, such as Newsbytes News Network (both were multiple winners); individual
columns and features by well-known journalists such as Steven Levy (author of “”); broadcast awards such as “Best Radio Program”; as well as book awards in
categories such as Best Product Specific Book.

 CPA President Jeff Yablon (1994-1996) developed an updated code of ethics for technology journalists that was adopted by many major trade show groups, most
notably Bruno Blenheim.

 The Computer Press Association disbanded in 2000.
</item 7>

<item 8>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 8>

<item 9>
Computer Magazine
Computer Magazine (and their website www.ComputerMagazine.com) is a popular magazine and online news site on computing and technology, offering current news and
reviews of popular and new business and consumer technologies, software, hardware, mobile computing, tablets, PCs, Macs, Windows, Linux, telecom, cellular,
wireless, data, cloud and science news on digital technologies and everything in the "tech-sphere and digi-verse", especially focused on information technology,
devices, software and services and related subjects, such as networking, servers, data centers and corporate data infrastructure technologies, and the Internet.

 Their online site, since 1997, is located at ComputerMagazine.com.

 "Computer Magazine" produces industry instructional and a popular ongoing webcast/podcast talk show and performs evaluations and reviews of IT industry
technology products, hardware, software and services with objective reporting widely respected as independent and objective, and trusted in the industry.
</item 9>

<item 10>
Sculptural Pursuit
Sculptural Pursuit was a quarterly art/literary magazine published by Hammer & Pen Productions, a Denver, Colorado publishing company.

 The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
</item 10>
=== END ITEM id=c53aa5fa87ae57cb ===

=== ITEM id=7fac816580a7b5b1 ===
Gold sentences and facts:
Sentence 0: Ecopack Ltd is a PET bottles blowing company in Pakistan.
  Fact 0.0: Ecopack Ltd is a PET bottles blowing company.
  Fact 0.1: Ecopack Ltd is located in Pakistan.
Sentence 1: Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and in combination with glass fiber for engineering resins.
  Fact 1.0: Polyethylene terephthalate is also written as poly(ethylene terephthalate).
  Fact 1.1: Polyethylene terephthalate is commonly abbreviated as PET.
  Fact 1.2: Polyethylene terephthalate is commonly abbreviated as PETE.
  Fact 1.3: Polyethylene terephthalate is sometimes abbreviated as PETP, which is obsolete.
  Fact 1.4: Polyethylene terephthalate is sometimes abbreviated as PET-P, which is obsolete.
  Fact 1.5: Polyethylene terephthalate is the most common thermoplastic polymer resin of the polyester family.
  Fact 1.6: Polyethylene terephthalate is used in fibers for clothing.
  Fact 1.7: Polyethylene terephthalate is used in containers for liquids and foods.
  Fact 1.8: Polyethylene terephthalate is used in thermoforming for manufacturing.
  Fact 1.9: Polyethylene terephthalate is used in combination with glass fiber for engineering resins.

Retrieved items:
<item 1>
Polycyclohexylenedimethylene terephthalate
Polycyclohexylenedimethylene terephthalate (PCT) is a thermoplastic polyester formed from the polycondensation of terephthalic acid and cyclohexanedimethanol.

Its chemical structure is similar to that of polyethylene terephthalate (PET), with which it shares properties like dimensional stability and chemical
resistance.

 PCT is also particularly resistant to high temperatures and hydrolysis.

 The melting point is 545 °F (285 °C).

 Common brand names are "Thermx" (Ticona), "Eastar" (Eastman) and "SkyPURA" (SK Chemicals).
</item 1>

<item 2>
Polyethylene 2,5-furandicarboxylate
Polyethylene 2,5-furandicarboxylate, also named poly(ethylene 2,5-furandicarboxylate), polyethylene furanoate and poly(ethylene furanoate) and generally
abbreviated as PEF, is a polymer that can be produced by polycondensation of 2,5-furandicarboxylic acid (FDCA) and monoethylene glycol (MEG).

 As an aromatic polyester from ethylene glycol it is a chemical analogue of polyethylene terephthalate (PET) and polyethylene naphthalate (PEN).

 PEF has been described in (patent) literature since 1951, but has gained renewed attention since the US department of energy proclaimed its building block,
FDCA, as a potential bio-based replacement for purified terephthalic acid (PTA) in 2004.
</item 2>

<item 3>
Two-liter bottle
The two-liter bottle is a common container for soft drinks.

 These bottles are produced from polyethylene terephthalate, also known as PET plastic, using the blow molding process.

 Bottle labels consist of a printed, tight-fitted plastic sleeve.

 A resealable screw-top allows the contents to be used at various times while retaining carbonation.
</item 3>

<item 4>
Polytrimethylene terephthalate
Polytrimethylene terephthalate (PTT), is a polyester synthesized and patented in 1941.

 It is produced by a method called condensation polymerization or transesterification.

 The two monomer units used in producing this polymer are: 1,3-propanediol and terephthalic acid or dimethyl terephthalate.

 Similar to polyethylene terephthalate, the PTT is used to make carpet fibers.
</item 4>

<item 5>
Krones
Krones AG is a German packaging and bottling machine manufacturer.

 It is the world's leading manufacturer of lines for filling beverages in plastic and glass bottles or beverage cans.

 The company manufactures stretch blow-moulding machines for producing polyethylene terephthalate (PET) bottles, plus fillers, labellers, bottle washers,
pasteurisers, inspectors, packers and palletisers.

 This product portfolio is complemented by material flow systems and process technology for producing beverages, plus syrup kitchens, for clients like
breweries, dairies and soft-drink companies.
</item 5>

<item 6>
Polyethylene terephthalate
Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common
thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and
in combination with glass fiber for engineering resins.
</item 6>

<item 7>
Xenoy
Xenoy is a blend of plastics with many industrially-useful properties.

 It is typically polyester (polybutylene terephthalate, PBT, or polyethylene terephthalate, PET) and polycarbonate (PC): it is often labeled PBT+PC or PET+PC.
</item 7>

<item 8>
BoPET
BoPET (Biaxially-oriented polyethylene terephthalate) is a polyester film made from stretched polyethylene terephthalate (PET) and is used for its high tensile
strength, chemical and dimensional stability, transparency, reflectivity, gas and aroma barrier properties, and electrical insulation.
</item 8>

<item 9>
Ecopack
Ecopack Ltd is a PET bottles blowing company in Pakistan.
</item 9>

<item 10>
Polyethylene naphthalate
Polyethylene naphthalate (poly(ethylene 2,6-naphthalate or PEN) is a polyester with good barrier properties (even better than Polyethylene terephthalate).

 Because it provides a very good oxygen barrier, it is particularly well-suited for bottling beverages that are susceptible to oxidation, such as beer.

 It is also used in making high performance sailcloth.

 It also has been found to show supreme scintillation properties and is expected to replace classic plastic scintillators.
</item 10>
=== END ITEM id=7fac816580a7b5b1 ===

=== ITEM id=31612306b6e41be5 ===
Gold sentences and facts:
Sentence 0: Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban parents in France, where she was also raised.
  Fact 0.0: Anaïs Nin was born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell.
  Fact 0.1: Anaïs Nin was born on February 21, 1903.
  Fact 0.2: Anaïs Nin died on January 14, 1977.
  Fact 0.3: Anaïs Nin was an essayist.
  Fact 0.4: Anaïs Nin was a memoirist.
  Fact 0.5: Anaïs Nin was born to Cuban parents.
  Fact 0.6: Anaïs Nin was born in France.
  Fact 0.7: Anaïs Nin was raised in France.
Sentence 1: Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as a major exponent of romantic modernism in Greece and the world.
  Fact 1.0: Odysseas Elytis is a pen name of Odysseas Alepoudellis.
  Fact 1.1: Odysseas Elytis's Greek name is Οδυσσέας Ελύτης.
  Fact 1.2: Odysseas Alepoudellis's Greek name is Οδυσσέας Αλεπουδέλλης.
  Fact 1.3: Odysseas Elytis was born on 2 November 1911.
  Fact 1.4: Odysseas Elytis died on 18 March 1996.
  Fact 1.5: Odysseas Elytis was regarded as a major exponent of romantic modernism in Greece.
  Fact 1.6: Odysseas Elytis was regarded as a major exponent of romantic modernism in the world.
Sentence 2:  In 1979 he was awarded the Nobel Prize in Literature.
  Fact 2.0: Odysseas Elytis was awarded the Nobel Prize in Literature in 1979.

Retrieved items:
<item 1>
Anaïs Nin: A Biography
Anaïs Nin: A Biography is Deirdre Bair's award-winning biography of writer Anaïs Nin.

 It is considered arguably by many to be the most comprehensive, well-researched, and scholarly biography available of Nin.

 Though the biography has received praise, it has also angered some fans of Anaïs Nin as well as some of her former associates, some of whom claim that Bair's
critical and rigorous investigation of Anaïs Nin's life is unsympathetic.
</item 1>

<item 2>
The Diary of Anaïs Nin
The Diary of Anaïs Nin is the published version of Anaïs Nin's own private manuscript diary, which she began at age 11 in 1914 during a trip from Europe to New
York with her mother and two brothers.

 Anaïs Nin would later say she had begun the diary as a letter to her father, Cuban composer Joaquín Nin, who had abandoned the family a few years earlier.
</item 2>

<item 3>
Naguib Mahfouz Medal for Literature
The Naguib Mahfouz Medal for Literature is a literary award for Arabic literature.

 It is given to the best contemporary novel written in Arabic, but not available in English translation.

 The winning book is then translated into English, and published by American University in Cairo Press.

 It was first awarded in 1996 and is presented annually on December 11, the birthday of Nobel laureate Naguib Mahfouz, by the President of the American
University in Cairo.
</item 3>

<item 4>
List of female Nobel laureates
The Nobel Prizes are awarded annually by the Royal Swedish Academy of Sciences, the Swedish Academy, the Karolinska Institute, and the Norwegian Nobel Committee
to individuals who make outstanding contributions in the fields of Chemistry, Physics, Literature, Peace, Physiology or Medicine and Economics.

 All but the economics prize were established by the 1895 will of Alfred Nobel, which dictates that the awards should be administered by the Nobel Foundation.

 The Nobel prize in Economics, or The Sveriges Riksbank Prize in Economic Sciences in Memory of Alfred Nobel, was established in 1968 by the Sveriges Riksbank,
the central bank of Sweden, for outstanding contributions in the field of Economics.

 Each prize is awarded by a separate committee; the Royal Swedish Academy of Sciences awards the Prizes in Physics, Chemistry, and Economics, the Swedish
Academy awards the Prize in Literature, the Karolinska Institute awards the Prize in Physiology or Medicine, and the Norwegian Nobel Committee awards the Prize
in Peace.

 Each recipient receives a medal, a diploma and a cash prize that has varied throughout the years.
</item 4>

<item 5>
Odysseas Elytis
Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as
a major exponent of romantic modernism in Greece and the world.

 In 1979 he was awarded the Nobel Prize in Literature.
</item 5>

<item 6>
Nearer the Moon
Nearer the Moon: From a Journal of Love (full title "Nearer the Moon: From a Journal of Love, the Unexpurgated Diary of Anaïs Nin (1937–1939)") is a 1996 book
based upon material excerpted from the unpublished diaries of Anaïs Nin.

 It corresponds temporally to part of Nin's published diaries.

 It consists mainly of material that was left out of the published version because it would have hurt people involved or their relationships with Anaïs Nin had
it been published at the time.
</item 6>

<item 7>
Nobel Prize in Literature
Since 1901, the Nobel Prize in Literature (Swedish: "Nobelpriset i litteratur" ) has been awarded annually to an author from any country who has, in the words
of the will of Alfred Nobel, produced "in the field of literature the most outstanding work in an ideal direction" (original Swedish: "den som inom litteraturen
har producerat det mest framstående verket i en idealisk riktning").

 Though individual works are sometimes cited as being particularly noteworthy, here "work" refers to an author's work as a whole.

 The Swedish Academy decides who, if anyone, will receive the prize in any given year.

 The academy announces the name of the chosen laureate in early October.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895; the others are the Nobel Prize in Chemistry, Nobel Prize in Physics, Nobel
Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 7>

<item 8>
Henry and June
Henry and June: From the Unexpurgated Diary of Anaïs Nin (full title "Henry and June: From A Journal of Love: the Unexpurgated Diary of Anaïs Nin (1931–1932)")
is a 1986 book that is based upon material excerpted from the unpublished diaries of Anaïs Nin.
</item 8>

<item 9>
Anaïs Nin
Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban
parents in France, where she was also raised.

 She spent some time in Spain and Cuba, but lived most of her life in the United States, where she became an established author.

 She wrote journals (which span more than 60 years, beginning when she was 11 years old and ending shortly before her death), novels, critical studies, essays,
short stories, and erotica.

 A great deal of her work, including "Delta of Venus" and "Little Birds", was published posthumously.
</item 9>

<item 10>
Incest: From a Journal of Love
Incest: From a Journal of Love: The Unexpurgated Diary of Anaïs Nin (1932–1934) is a 1992 non-fiction book by Anaïs Nin.

 It is a continuation of the diary entries first published in "Henry and June: From the Unexpurgated Diary of Anaïs Nin".

 It features Nin's relationships with writer Henry Miller, his wife June Miller, the psychoanalyst Otto Rank, her father Joaquín Nin, and her husband Hugh
Parker Guiler.

 She also copied some of her correspondence with these people into her diary.

 Much of this book was written in English, although those of her letters which were originally written in French and Spanish were translated.

 Most of this diary takes place in France, particularly Clichy, Paris and Louveciennes.
</item 10>
=== END ITEM id=31612306b6e41be5 ===

=== ITEM id=ff498a93db1a8269 ===
Gold sentences and facts:
Sentence 0: Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.
  Fact 0.0: Eucryphia is a small genus.
  Fact 0.1: Eucryphia consists of trees and large shrubs.
  Fact 0.2: Eucryphia is native to the south temperate regions of South America.
  Fact 0.3: Eucryphia is native to coastal eastern Australia.
Sentence 1:  Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.
  Fact 1.0: Eucryphia is sometimes placed in a family of its own, the Eucryphiaceae.
  Fact 1.1: More recent classifications place Eucryphia in the Cunoniaceae.
Sentence 2:  There are seven species, two in South America and five in Australia, and several named hybrids.
  Fact 2.0: Eucryphia has seven species.
  Fact 2.1: Two Eucryphia species are in South America.
  Fact 2.2: Five Eucryphia species are in Australia.
  Fact 2.3: Eucryphia has several named hybrids.
Sentence 3: Lepidozamia is a genus of two species of cycad, both endemic to Australia.
  Fact 3.0: Lepidozamia is a genus of cycad.
  Fact 3.1: Lepidozamia has two species.
  Fact 3.2: Both Lepidozamia species are endemic to Australia.

Retrieved items:
<item 1>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 1>

<item 2>
Lepidozamia comprises two Australian rainforest cycads (2n=18). Orange Field, a Sri Lankan tea factory founded in 1950, makes about 500,000 kg of black tea
annually for Middle Eastern and CIS markets, sourcing from 750 smallholders. Lithocarpus has 334 accepted East/Southeast Asian species; three Java endemics
listed here are endangered, while L. glaber occurs in Japan, China and Taiwan. Oaks (Querc
</item 2>

<item 3>
Eucryphia: seven mostly evergreen South American/Australian trees; classified in Cunoniaceae. Bothriochloa ischaemum: widespread grass; var. songarica invasive
in Texas. Eriogonum buckwheats: genus diverse across North America; E. heracleoides inhabits western U.S.; E. diclinum forms low mats on serpentine soils; E.
crosbyae occurs in Oregon/Nevada.
</item 3>

<item 4>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 4>

<item 5>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 5>

<item 6>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 6>

<item 7>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 7>

<item 8>
Leitneria
Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United
States states of Arkansas, Florida, Georgia, Missouri and Texas.
</item 8>

<item 9>
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 9>

<item 10>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 10>
=== END ITEM id=ff498a93db1a8269 ===

=== ITEM id=a66d016107c4a585 ===
Gold sentences and facts:
Sentence 0: Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on science and technology subjects.
  Fact 0.0: Popular Science is also known as PopSci.
  Fact 0.1: Popular Science is an American bi-monthly magazine.
  Fact 0.2: Popular Science carries popular science content.
  Fact 0.3: Popular science content refers to articles for the general reader on science and technology subjects.
Sentence 1: Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.
  Fact 1.0: Girlfriends was a women's magazine.
  Fact 1.1: Girlfriends provided critical coverage of culture.
  Fact 1.2: Girlfriends provided critical coverage of entertainment.
  Fact 1.3: Girlfriends provided critical coverage of world events.
  Fact 1.4: Girlfriends' coverage was from a lesbian perspective.
Sentence 2:  Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.
  Fact 2.0: Girlfriends was published monthly from San Francisco since 1993.
  Fact 2.1: Girlfriends was distributed nationwide by Disticor.
Sentence 3:  "Girlfriends" magazine ceased publication in 2006.
  Fact 3.0: Girlfriends magazine ceased publication in 2006.

Retrieved items:
<item 1>
Jack and Jill (magazine)
Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.

 It features stories and educational activities.
</item 1>

<item 2>
Popular science
Popular science (also pop-science or popsci) is interpretation of science intended for a general audience.

 While science journalism focuses on recent scientific developments, popular science is more broad-ranging.

 It may be written by professional science journalists or by scientists themselves.

 It is presented in many forms, including books, film and television documentaries, magazine articles, and web pages.
</item 2>

<item 3>
Popular Science
Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on
science and technology subjects.

 "Popular Science" has won over 58 awards, including the American Society of Magazine Editors awards for its journalistic excellence in both 2003 (for General
Excellence) and 2004 (for Best Magazine Section).

 With roots beginning in 1872, "Popular Science" has been translated into over 30 languages and is distributed to at least 45 countries.
</item 3>

<item 4>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 4>

<item 5>
Randy J. Goodwin
Randy J. Goodwin, born as Ivan Jerome Goodwin, is a Hollywood actor.

 He is perhaps best known for his role as "Kennedy Winslow" on "Fast Track" and "Max Ellis" on the show "Abby", as well as his recurring roles on "Girlfriends"
as "Davis Hamilton" and "The Vampire Diaries" as "Dr. Jonas Martin".
</item 5>

<item 6>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 6>

<item 7>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 7>

<item 8>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 8>

<item 9>
David Bradley (UK journalist)
David Bradley (born 1966 in England) is a British journalist specializing in science and technology.

 After graduating in 1988 with a degree in Chemistry from Newcastle University, he began his career in technical editing at the Royal Society of Chemistry in
1989 and built up a freelance writing business in his spare time before going full-time freelance in the mid-1990s.

 He has contributed to a wide range of popular science publications, including "Popular Science", "American Scientist", "New Scientist" and "Science".

 As well as numerous newspapers ("The Guardian", "The Daily Telegraph") and trade magazines ("Chemistry in Britain", "Proceedings of the National Academy of
Sciences of the United States of America", "Nature"), and websites (ChemWeb.com, BioMedNet.com, SpectroscopyNOW.com).
</item 9>

<item 10>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 10>
=== END ITEM id=a66d016107c4a585 ===

=== ITEM id=d629827d67264c26 ===
Gold sentences and facts:
Sentence 0: McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.
  Fact 0.0: McFarland, USA is also known as McFarland.
  Fact 0.1: McFarland, USA is a 2015 American sports drama film.
  Fact 0.2: McFarland, USA was directed by Niki Caro.
  Fact 0.3: McFarland, USA was produced by Mark Ciardi.
  Fact 0.4: McFarland, USA was produced by Gordon Gray.
  Fact 0.5: McFarland, USA was written by Christopher Cleveland.
  Fact 0.6: McFarland, USA was written by Bettina Gilois.
  Fact 0.7: McFarland, USA was written by Grant Thompson.
  Fact 0.8: McFarland, USA has music composed by Antônio Pinto.
Sentence 1: The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.
  Fact 1.0: The Black Hole is a 1979 American science fiction film.
  Fact 1.1: The Black Hole was directed by Gary Nelson.
  Fact 1.2: The Black Hole was produced by Walt Disney Productions.

Retrieved items:
<item 1>
The Black Hole
The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.

 The film stars Maximilian Schell, Robert Forster, Joseph Bottoms, Yvette Mimieux, Anthony Perkins, and Ernest Borgnine, while the voices of the main robot
characters are provided by Roddy McDowall and Slim Pickens (both unbilled).

 The music for the film was composed by John Barry.

 It was the first film from Walt Disney Productions to receive a PG rating.

 The film was released on December 18, 1979 in the United Kingdom and on December 21, 1979 in the United States.
</item 1>

<item 2>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 2>

<item 3>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 3>

<item 4>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 4>

<item 5>
Full Metal Jacket
Full Metal Jacket is a 1987 British-American war film directed and produced by Stanley Kubrick.

 The screenplay by Kubrick, Michael Herr, and Gustav Hasford was based on Hasford's novel "The Short-Timers" (1979).

 Its storyline follows a platoon of U.S. Marines through their training, primarily focusing on two privates, Joker and Pyle, who struggle to get through camp
under their foul-mouthed drill instructor, Gunnery Sergeant Hartman, and the experiences of two of the platoon's Marines in the Tet Offensive during the Vietnam
War.

 The film's title refers to the full metal jacket bullet used by soldiers.

 The film was released in the United States on June 26, 1987.
</item 5>

<item 6>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 6>

<item 7>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 7>

<item 8>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 8>

<item 9>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 9>

<item 10>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 10>
=== END ITEM id=d629827d67264c26 ===

=== ITEM id=15ba50383a28d7c4 ===
Gold sentences and facts:
Sentence 0: The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver Division, the Shenango Division, and the Conneaut Division.
  Fact 0.0: The Beaver and Erie Canal was also known as the Erie Extension Canal.
  Fact 0.1: The Beaver and Erie Canal was part of the Pennsylvania Canal system.
  Fact 0.2: The Beaver and Erie Canal consisted of three sections.
  Fact 0.3: The three sections of the Beaver and Erie Canal were the Beaver Division, the Shenango Division, and the Conneaut Division.
Sentence 1: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.
  Fact 1.0: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp.
  Fact 1.1: The Dismal Swamp Canal is located in Virginia and North Carolina.
  Fact 1.2: Virginia and North Carolina are in the United States.

Retrieved items:
<item 1>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 1>

<item 2>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
</item 2>

<item 3>
The entries cover historic U.S. canals, waterways, water systems, and sites. Transport canals include Beaver–Erie (136 miles), Delaware (60), Whitewater (76),
and Susquehanna–Tidewater (43), plus interstate and regional routes; many declined with railroad competition. The Dismal Swamp Canal, opened in 1805 and closed
in 2016, served the Intracoastal Waterway; nearby park and refuge protect the swamp.
</item 3>

<item 4>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 4>

<item 5>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 5>

<item 6>
Dismal Swamp State Park
Dismal Swamp State Park is a North Carolina state park in Camden County, North Carolina in the United States.

 The park was created as a state natural area in 1974 with the help of The Nature Conservancy, and on July 28, 2007 the NC General Assembly re-designated it as
a state park.

 It opened to the public in 2008.

 This marked the first time that public access to Great Dismal Swamp was made possible in North Carolina.

 The park covers 14432 acre of protected land on the North Carolina/Virginia border.

 Park offices are three miles (5 km) south of the border on U.S. Route 17 near South Mills.

 Features of the park include the canal which is used regularly by boaters using the Intracoastal Waterway and several miles of hiking and biking trails.
</item 6>

<item 7>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 7>

<item 8>
Great Dismal Swamp National Wildlife Refuge
The Great Dismal Swamp National Wildlife Refuge was created in 1974 to help protect and preserve a portion of the Great Dismal Swamp, a marshy region on the
Coastal Plain of southeastern Virginia and northeastern North Carolina between Norfolk, Virginia, and Elizabeth City, North Carolina in the United States.

 It is located in parts of the independent cities of Chesapeake and Suffolk in Virginia, and the counties of Camden, Gates, and Pasquotank in North Carolina.
</item 8>

<item 9>
New York’s Erie Canal heritage includes Lock 52 (1849–53), canal buildings and archaeological sites; the 1836–42 Broad Street Aqueduct later became a road and
subway route. The 1905 song “Low Bridge” recalls mule-powered canal traffic. The 525-mile state canal system includes the Erie, Oswego, Cayuga–Seneca and
Champlain canals. The Genesee River and tributaries Black Creek and Oatka Creek drain western New
</item 9>

<item 10>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 10>
=== END ITEM id=15ba50383a28d7c4 ===

=== ITEM id=67521214576f6fa4 ===
Gold sentences and facts:
Sentence 0: Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.
  Fact 0.0: Isaac Asimov was born Isaak Ozimov.
  Fact 0.1: Isaac Asimov was born on January 2, 1920.
  Fact 0.2: Isaac Asimov's dates run from January 2, 1920 to April 6, 1992.
  Fact 0.3: Isaac Asimov was a Russian-American writer.
  Fact 0.4: Isaac Asimov was a professor of biochemistry at Boston University.
Sentence 1: Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.
  Fact 1.0: Michael Cunningham was born on November 6, 1952.
  Fact 1.1: Michael Cunningham is a U.S. novelist.
  Fact 1.2: Michael Cunningham is a U.S. screenwriter.

Retrieved items:
<item 1>
Isaac Asimov's Robot City
Isaac Asimov's Robot City is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 It takes place between "The Robots of Dawn" and "Robots and Empire".

 Each volume is complete in itself, but they form a continuing series.

 The novels were written in response to a writing challenge issued by Asimov to write a series involving the Three Laws of Robotics, which brought about a
collaboration of several authors.

 Asimov provided outlines for stories which filled in the gap between Asimov's own robot stories and his "Foundation" series, explaining the disappearance of
the robots prior to the establishment of the galactic empire. "

Isaac Asimov's Robots and Aliens" followed in this series, with the same protagonists and many other characters.

 The common theme of all books of both series is the interaction between the characters and autonomous cities run and populated by robots (the "robot cities" of
the series title).

 Robot City was also released as a mystery game for the PC in 1995.

 The player takes the role of Derec.
</item 1>

<item 2>
Wandering Stars
Wandering Stars is an anthology of Jewish fantasy and science fiction, edited by American writer Jack Dann, originally published by Harper & Row in 1974.

 It represented, according to the book cover, "the first time in science fiction that the Jew - and the richness of his themes and particular points of view --
will appear without a mask."

 In his introduction, "Why Me?"

, Isaac Asimov discussed how many Jewish science fiction writers prior to that time had used gentile pen names in order to get published: "Many of the Jewish
pulp writers, however, used pen names as a matter of sound business.

 A story entitled "War Gods of the Oyster-Men of Deneb" did not carry conviction if it was written by someone named Chaim Itzkowitz."

 He then goes on to discuss the pen names of various Jewish writers included in this book.

 "Wandering Stars" is therefore of historical significance as the first science fiction anthology where Jewish writers openly identified themselves as such.

 It was followed by a second anthology, "More Wandering Stars", also edited by Jack Dann, published by Doubleday in 1981.
</item 2>

<item 3>
Isaac Asimov's Robot City: Perihelion
Isaac Asimov's Robot City: Perihelion is a book written in 1988 by William F. Wu.

 It is part of the series Isaac Asimov's Robot City, which was inspired by Isaac Asimov's "Robot" series.
</item 3>

<item 4>
Isaac Asimov's Robot City: Odyssey
Isaac Asimov's Robot City: Odyssey is a young adult science fiction novel written in 1987 by Michael P. Kube-McDowell.

 It is part of the series Isaac Asimov's Robot City, inspired by Isaac Asimov's "Robot" series.

 The 1995 computer game Robot City is based on the plot of "Odyssey".
</item 4>

<item 5>
Witches (anthology)
Witches is an anthology of themed fantasy and science fiction short stories on the subject of witches edited by Isaac Asimov, Martin H. Greenberg and Charles G.
Waugh as the second volume in their Isaac Asimov's Magical Worlds of Fantasy series.

 It was first published in paperback by Signet/New American Library in April 1984.

 It was later gathered together with "Wizards", the first book in the series, into the omnibus hardcover collection "Isaac Asimov's Magical Worlds of Fantasy:
Witches & Wizards" (1985).
</item 5>

<item 6>
Isaac Asimov's Robots and Aliens
Isaac Asimov's Robots and Aliens is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 Each volume is complete in itself, but they form a continuing series.

 The series follows the action of the novels of the "Isaac Asimov's Robot City" series, with the same protagonists Derec and Ariel, and many other characters.

 The plot deals with the Three Laws and encounters between robots and different varieties of alien life.
</item 6>

<item 7>
The Last Question
"The Last Question" is a science fiction short story by American writer Isaac Asimov.

 It first appeared in the November 1956 issue of "Science Fiction Quarterly" and was anthologized in the collections "Nine Tomorrows" (1959), "The Best of Isaac
Asimov" (1973), "Robot Dreams" (1986), the retrospective "Opus 100" (1969), and in "Isaac Asimov: The Complete Stories, Vol.

 1" (1990).

 It was Asimov's favorite short story of his own authorship, and is one of a loosely connected series of stories concerning a fictional computer called
Multivac.

 The story overlaps science fiction, theology, and philosophy.
</item 7>

<item 8>
Isaac Asimov
Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.

 He was known for his works of science fiction and popular science.

 Asimov was a prolific writer, and wrote or edited more than 500 books and an estimated 90,000 letters and postcards. His books have been published in 9 of the
10 major categories of the Dewey Decimal Classification.
</item 8>

<item 9>
Wizards (Asimov anthology)
Wizards is an anthology of themed fantasy and science fiction short stories on the subject of wizards edited by Isaac Asimov, Martin H. Greenberg and Charles G.
Waugh.

 The first volume in their Isaac Asimov's Magical Worlds of Fantasy series, it was first published in paperback by Signet/New American Library in October 1983.

 It was later gathered together with "Witches", the second book in the series, into the omnibus hardcover collection "Isaac Asimov's Magical Worlds of Fantasy:
Witches & Wizards" (1985).
</item 9>

<item 10>
Isaac Asimov's Magical Worlds of Fantasy
Isaac Asimov's Magical Worlds of Fantasy is a series of twelve themed paperback fantasy and science fiction anthologies edited by Isaac Asimov, Martin H.
Greenberg and Charles G. Waugh, a companion set to the ten volume "Isaac Asimov's Wonderful Worlds of Science Fiction", produced by the same editors.

 It was published by Signet/New American Library from 1983 to 1991.

 Volumes 1 and 2 were also issued in hardcover in an omnibus collection titled "Isaac Asimov's Magical Worlds of Fantasy: Witches & Wizards".
</item 10>
=== END ITEM id=67521214576f6fa4 ===

=== ITEM id=7a50832fe857fe9e ===
Gold sentences and facts:
Sentence 0:  He is most well known for providing the singing voice of Tony (played onscreen by Richard Beymer) in the 1961 film musical "West Side Story".
  Fact 0.0: This person is most well known for providing the singing voice of Tony in the 1961 film musical "West Side Story".
  Fact 0.1: Richard Beymer played Tony onscreen in the 1961 film musical "West Side Story".
Sentence 1:  The film was nominated for 11 Academy Awards and won 10, including Best Picture (as well as a special award for Robbins), becoming the record holder for the most wins for a movie musical.
  Fact 1.0: West Side Story was nominated for 11 Academy Awards.
  Fact 1.1: West Side Story won 10 Academy Awards.
  Fact 1.2: West Side Story won Best Picture.
  Fact 1.3: West Side Story won a special award for Robbins.
  Fact 1.4: West Side Story became the record holder for the most wins for a movie musical.

Retrieved items:
<item 1>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 1>

<item 2>
Academy Honorary Award
The Academy Honorary Award – instituted in 1948 for the 21st Academy Awards (previously called the Special Award, which was first presented in early 1929) – is
given annually by the Board of Governors of the Academy of Motion Picture Arts and Sciences (AMPAS) to celebrate motion picture achievements that are not
covered by existing Academy Awards, although prior winners of competitive Academy Awards are not excluded from receiving the Honorary Award.

 Unless otherwise specified, Honorary Award recipients receive the same gold Oscar statuettes received by winners of the competitive Academy Awards.

 Unlike the Special Achievement Award instituted in 1972 (and discontinued in 1995), those on whom the Academy confers its Honorary Award do not have to meet
"the Academy's eligibility year and deadline requirements."

 Like the Special Achievement Award, the Special Award and Honorary Award have been used to reward significant achievements of the year that did not fit in
existing categories, subsequently leading the Academy to establish several new categories, and to honor exceptional career achievements, contributions to the
motion picture industry, and service to the Academy.
</item 2>

<item 3>
Triple Crown of Acting
The Triple Crown of Acting is a term used in the American entertainment industry to describe actors who have won a competitive Academy Award, Emmy Award, and
Tony Award in the acting categories.

 23 people have achieved the triple crown of acting (14 women, 9 men).

 Helen Hayes' Emmy Award win on February 5, 1953, made her the first person to achieve the triple crown.

 Thomas Mitchell became the first man to achieve the triple crown with his Tony Award win later the same year on March 29, 1953.

 Hayes and Rita Moreno are the only triple crown winners in competitive acting categories who have also won a Grammy Award to complete the EGOT.
</item 3>

<item 4>
Jimmy Bryant (singer)
James Howard Bryant (born June 2, 1929) is a singer, arranger and composer.

 He is most well known for providing the singing voice of Tony (played onscreen by Richard Beymer) in the 1961 film musical "West Side Story".

 While he received no screen credit, he states that Beymer was "a nice guy, and every time he did an interview he would mention my name."

 He also sang for James Fox in the 1967 film musical "Thoroughly Modern Millie", and sang in "The Telephone Hour" number in "Bye Bye Birdie".

 He also sang in the group that performed the theme song of the TV series "Batman".
</item 4>

<item 5>
Sing (2016 American film)
Sing is a 2016 American 3D computer-animated musical comedy film produced by Illumination Entertainment.

 It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth
MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.

 The film is about a group of anthropomorphic animals that enter a singing competition, hosted by a koala hoping to save his theater.
</item 5>

<item 6>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 6>

<item 7>
Geoffrey Rush
Geoffrey Roy Rush {'1': ", '2': ", '3': ", '4': "} (born 6 July 1951) is an Australian actor and film producer.

 Rush is the youngest amongst the few people who have won the "Triple Crown of Acting": the Academy Award, the Primetime Emmy Award, and the Tony Award.

 He has won one Academy Award for acting (from four nominations), three British Academy Film Awards (from five nominations), two Golden Globe Awards and four
Screen Actors Guild Awards. Rush is the founding President of the Australian Academy of Cinema and Television Arts and was named the 2012 Australian of the
Year.

 He is also the first actor to win the Academy Award, BAFTA Award, Critics' Choice Movie Award, Golden Globe Award, and Screen Actors Guild Award for a single
performance in film for his performance in "Shine" (1996).
</item 7>

<item 8>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 8>

<item 9>
Academy Honorary Award
The Academy Honorary Award is often awarded in preference to those with noted achievements in motion pictures who have nevertheless never won an Academy Award.

 Thus, many of its recipients are Classic Hollywood stars, such as Lillian Gish, Barbara Stanwyck, Kirk Douglas, and Lauren Bacall.
</item 9>

<item 10>
The Godfather (soundtrack)
The Godfather is the soundtrack from the film of the same name, released in 1972 by Paramount Records, and in 1991 on compact disc by MCA.

 Unless noted, the cues were composed by Nino Rota and conducted by Carlo Savina (who was credited on the LP, but not the CD).

 The song "I Have but One Heart" is sung by Al Martino, who performed it in the film as character Johnny Fontane.
</item 10>
=== END ITEM id=7a50832fe857fe9e ===

=== ITEM id=d3c3a8800ebd8dc2 ===
Gold sentences and facts:
Sentence 0: Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming pieces and a game system.
  Fact 0.0: Icehouse pieces are also called Icehouse Pyramids.
  Fact 0.1: Treehouse pieces are also called Treehouse Pyramids.
  Fact 0.2: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, and Treehouse Pyramids are officially called Looney Pyramids.
  Fact 0.3: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are nestable pyramid-shaped gaming pieces.
  Fact 0.4: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are stackable pyramid-shaped gaming pieces.
  Fact 0.5: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are a game system.
Sentence 1:  The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
  Fact 1.0: The Looney Pyramids game system was invented by Andrew Looney.
  Fact 1.1: The Looney Pyramids game system was invented by John Cooper.
  Fact 1.2: The Looney Pyramids game system was invented in 1987.
  Fact 1.3: The Looney Pyramids game system was originally for use in the game of Icehouse.
Sentence 2:  In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
  Fact 2.0: "Kill Doctor Lucky" won the Origins Award in 1998.
  Fact 2.1: "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".

Retrieved items:
<item 1>
Treehouse (game)
Treehouse is a game in which players try to get their configuration of Icehouse pieces to match the central configuration, shared by all players.

 The rolling of the special "Treehouse Die" tells the player what kind of move to make to change his own or the central configuration, and then he does so to
best move towards the goal.
</item 1>

<item 2>
Icehouse (song)
"Icehouse" is a song by the Australian rock band Flowers.

 It was released as a single in Europe in 1982 by Chrysalis Records from the band's first album, "Icehouse", after the band changed its name to Icehouse.

 In the United States, the song peaked at number 28 on the "Billboard" Top Tracks chart in 1981.
</item 2>

<item 3>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 3>

<item 4>
Icehouse’s 1985 album *Boxes*, composed by Iva Davies and Robert Kretschmer for Graeme Murphy’s Sydney Dance Company ballet, premiered at the Sydney Opera House
and was issued in the US as *Sydney Dance Company*. Its single “No Promises” reached #30 in Australia; two album tracks later appeared on *Measure for Measure*.
Earlier Flowers/Icehouse singles included “Can’t Help Myself” (#10), “We Can Get Together” (#16),
</item 4>

<item 5>
Can't Help Myself
"Can't Help Myself" is the first single released by the Australian synthpop/rock band Flowers, later known as Icehouse.

 It was released in May 1980 as a 7" vinyl single on independent label, Regular Records, five months ahead of debut album "Icehouse".

 A 10" vinyl single was released in July and had a cover depicting individual images of band members diagonally across the band's name and the single's title
(see infobox at right middle).

 It peaked at #10 on the Australian Singles Charts.
</item 5>

<item 6>
Icehouse pieces
Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming
pieces and a game system.

 The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
</item 6>

<item 7>
The entries cover games across several eras and genres: Japan-only 1989 Famicom game *A Week of Garfield*, based on the comic; *Kill Doctor Lucky*, a humorous
1996 board game that won a 1997 Origins Award; Sid Sackson’s dice-risk racing game *Can’t Stop* (1980; reprinted 2007, iOS 2012); 2003 light war game *Attack!*,
also an Origins Award winner; the luck-based
</item 7>

<item 8>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 8>

<item 9>
Icehouse pyramids are stackable game pieces invented by Andrew Looney and John Cooper in 1987; Treehouse uses a die to guide moves matching players’
configurations to a shared one. Awithlaknannai Mosona is a Zuni Alquerque-related capture game documented in 1907. Alhambra (2003) is Dirk Henn’s Arabian-themed
tile game. Brunei’s Pasang scores captured tokens; Punjab’s Sher-bakar is a tiger hunt
</item 9>

<item 10>
Boxes (Icehouse album)
Boxes is a soundtrack album by Australian band Icehouse, released by Festival Records / Chrysalis Records in November 1985.

 The work was originally conceived by its composers, Iva Davies and Robert Kretschmer, in collaboration with choreographer Graeme Murphy of the Sydney Dance
Company for performance as the ballet "Boxes".

 The first live performance of "Boxes" was given by the Sydney Dance Company together with Davies and Kretschmer of Icehouse and guest percussionist Masaki
Tanazawa in the Opera Theatre of the Sydney Opera House on 7 November 1985.

 "Boxes" was released in the US under the name Sydney Dance Company.

 The single "No Promises" was released as a 12" in November and a cassingle in December on Regular Records for Australian and New Zealand markets, it contained
other tracks from "Boxes"; two tracks from "Boxes" were later released on their next studio album "Measure for Measure" in April 1986.
</item 10>
=== END ITEM id=d3c3a8800ebd8dc2 ===
