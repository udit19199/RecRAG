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

This file has 11 items.

=== ITEM id=aa5c978db8f4bf04 ===
Gold sentences and facts:
Sentence 0: Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and Vietnam.
  Fact 0.0: Aspidistra is a genus of flowering plants.
  Fact 0.1: Aspidistra is in the family Asparagaceae.
  Fact 0.2: Aspidistra is in the subfamily Nolinoideae.
  Fact 0.3: Aspidistra is native to eastern Asia.
  Fact 0.4: Aspidistra is native to southeastern Asia.
  Fact 0.5: Aspidistra is particularly native to China.
  Fact 0.6: Aspidistra is particularly native to Vietnam.
Sentence 1: Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
  Fact 1.0: Cyrtanthus is a genus of perennial plants.
  Fact 1.1: Cyrtanthus is a genus of herbaceous plants.
  Fact 1.2: Cyrtanthus is a genus of bulbous plants.
  Fact 1.3: Cyrtanthus is in the family Amaryllidaceae.
  Fact 1.4: Cyrtanthus is in the subfamily Amaryllidoideae.

Retrieved items:
<item 1>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 1>

<item 2>
Amarylloidinae
Amarylloidinae is a now obsolete informal name for an "infrafamily" within the Amaryllidaceae (Amaryllis) family, erected by Hamilton Traub.

 This grouping was designed to fill a perceived gap between the formal rank of subfamily and tribes.

 In his treatment of this family, he divided it first into four subfamiles.

 Within subfamily Amarylloideae he then divided his sixteen tribes into two infrafamilies, Amarylloidinae (12 tribes) and Pancratioidinae (4 tribes), both of
which were subsequently demonstrated to be polyphyletic, and hence were abandoned by Dahlgren, who used no rank between family and tribe.

 On the other hand, he also used a much more restricted Amaryllidaceae corresponding to Traub's subfamily Amarylloideae.

 Thus Traub's Amarylloideae most closely resembles subfamily Amaryllidoideae "sensu" APGIII.
</item 2>

<item 3>
Aspidistra
Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and
Vietnam.

 They grow in shade under trees and shrubs.

 Their leaves arise more or less directly from ground level, where their flowers also appear.

 The number of species known has increased considerably from the 1980s onwards, with around 100 accepted as of July 2013 . "

Aspidistra elatior" is common worldwide as a foliage house plant that is very tolerant of neglect.

 It and other species can also be grown in shade outside, where they are generally hardy to -5 C .
</item 3>

<item 4>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 4>

<item 5>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 5>

<item 6>
Ammocharis
Ammocharis is a small genus from sub-Saharan Africa, in the Amaryllidaceae family (subfamily Amaryllidoideae) which includes 7 species distributed in Africa.

 The plant grows as a succulent, above-ground bulb, preferring seasonally wet, hot, sandy soils and full sun.
</item 6>

<item 7>
Graph query row:
entity: Cyrtanthus
relationship: PART_OF
related: Amaryllidoideae
</item 7>

<item 8>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 8>

<item 9>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 9>

<item 10>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 10>
=== END ITEM id=aa5c978db8f4bf04 ===

=== ITEM id=0ca93d193e3ff499 ===
Gold sentences and facts:
Sentence 0: Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international management consultant.
  Fact 0.0: Eenasul Fateh was born on 3 April 1959.
  Fact 0.1: Eenasul Fateh's Bengali name is ঈনাসুল ফাতেহ.
  Fact 0.2: Eenasul Fateh is also known by his stage name Aladin.
  Fact 0.3: Eenasul Fateh is a Bangladeshi-British cultural practitioner.
  Fact 0.4: Eenasul Fateh is a magician.
  Fact 0.5: Eenasul Fateh is a live artist.
  Fact 0.6: Eenasul Fateh is a former international management consultant.
Sentence 1: Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational problems and the development of plans for improvement.
  Fact 1.0: Management consulting is the practice of helping organizations improve their performance.
  Fact 1.1: Management consulting operates primarily through the analysis of existing organizational problems.
  Fact 1.2: Management consulting operates primarily through the development of plans for improvement.

Retrieved items:
<item 1>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 1>

<item 2>
Eenasul Fateh
Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live
artist and former international management consultant.
</item 2>

<item 3>
Arabian Nights and Days
Arabian Nights and Days (1979) is a novel by Egyptian writer Naguib Mahfouz, winner of the Nobel Prize for Literature.

 The novel serves as a sequel and companion piece for "One Thousand and One Nights" and includes many of the same characters that appeared in the original work
such as Shahryar, Scheherazade, and Aladdin.
</item 3>

<item 4>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 4>

<item 5>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 5>

<item 6>
Lawrence A. Gordon
Lawrence A. Gordon is the EY Alumni Professor of Managerial Accounting and Information Assurance at the University of Maryland’s Robert H. Smith School of
Business.

 He is also an Affiliate Professor in the University of Maryland Institute for Advanced Computer Studies.

 Dr. Gordon earned his Ph.D. in Managerial Economics from Rensselaer Polytechnic Institute.

 An internationally known scholar in the areas of managerial accounting (often called management accounting) and cybersecurity economics, Dr. Gordon's research
focuses on such issues as economic aspects of information security (including cybersecurity or computer security), corporate performance measures, cost
management systems, and capital investments.

 He is the author of approximately 100 articles, published in such journals as The Accounting Review, ACM Transactions on Information and System Security,
Communications of the ACM, Journal of Financial and Quantitative Analysis, Journal of Computer Security, MIS Quarterly, Accounting, Organizations and Society,
and the Journal of Accounting and Public Policy.

 Dr. Gordon's current research emphasizes the importance of applying concepts from economics and managerial accounting to an information-based economy.

 Dr.
</item 6>

<item 7>
Alex Algard
Alex Algard is an Internet entrepreneur.

 He is the founder and former CEO of CarDomain, the founder and former CEO of Whitepages.com, and the founder and current CEO of Hiya.
</item 7>

<item 8>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 8>

<item 9>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 9>

<item 10>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 10>
=== END ITEM id=0ca93d193e3ff499 ===

=== ITEM id=50aa9da86a5d3917 ===
Gold sentences and facts:
Sentence 0: Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with punctate stigmas on the female flowers.
  Fact 0.0: Lithocarpus is a genus in the beech family Fagaceae.
  Fact 0.1: Lithocarpus differs from "Quercus" in the erect spikes of insect-pollinated male flowers.
  Fact 0.2: Lithocarpus differs from "Quercus" in the short styles with punctate stigmas on the female flowers.
Sentence 1:  It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.
  Fact 1.0: Lithocarpus contains 17 species of shrubs and small trees.
  Fact 1.1: The Lithocarpus species are native from southern Florida to Mexico and South America.

Retrieved items:
<item 1>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 1>

<item 2>
Lithocarpus crassinervius
Lithocarpus crassinervius is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 2>

<item 3>
Lithocarpus glaber
Lithocarpus glaber, the Japanese oak, is a tree species in the genus "Lithocarpus" found in Japan, China and Taïwan.
</item 3>

<item 4>
Lophodermium
Lophodermium is a genus of fungi within the family Rhytismataceae.

 The genus contains 145 species and has a global distribution.

 Species of this genus are usually observed producing zone lines, conidiomata and ascomata on dead fallen leaves, but at least some are known to colonize living
leaves.

 In many cases they then live inside the colonized leaf as a symptomless endobiont, where they are regarded as detritivores utilising dead plant matter.

 In a few cases they may kill all or part of the leaf prematurely, and there is a substantial literature dealing with those species as plant pathogens.

 The genus infects many different plant families but with a notable concentration in the family Pinaceae; many "Lophodermium" species are restricted to a single
host genus (or even species), but some, particularly those infecting grasses, may infect several genera.

 Some are economically important plant pathogens, such as those that cause needlecast disease in European Black Pine, Scots Pine and Red Pine in forestry and
christmas tree plantations.
</item 4>

<item 5>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 5>

<item 6>
Lithocarpus kostermansii
Lithocarpus kostermansii is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 6>

<item 7>
Lithocarpus platycarpus
Lithocarpus platycarpus is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.

.
</item 7>

<item 8>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 8>

<item 9>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 9>

<item 10>
Duranta erecta
Duranta erecta is a species of flowering shrub in the verbena family Verbenaceae, native from Mexico to South America and the Caribbean.

 It is widely cultivated as an ornamental plant in tropical and subtropical gardens throughout the world, and has become naturalized in many places.

 It is considered an invasive species in Australia, China, South Africa and on several Pacific Islands.

 Common names include golden dewdrop, pigeon berry, and skyflower.

 In Mexico, the native Nahuatl name for the plant is xcambocoché.

 In Tonga it is known as mavaetangi (tears of departure).
</item 10>
=== END ITEM id=50aa9da86a5d3917 ===

=== ITEM id=bc1e6feb28443b84 ===
Gold sentences and facts:
Sentence 0:  He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth Games in Edinburgh, Scotland.
  Fact 0.0: He also participated in the 10,000 metres at the 1967 Pan American Games.
  Fact 0.1: The 1967 Pan American Games were held in Winnipeg, Manitoba, Canada.
  Fact 0.2: He represented Guyana at the 1970 British Commonwealth Games.
  Fact 0.3: The 1970 British Commonwealth Games were held in Edinburgh, Scotland.
Sentence 1:  The race consists of 25 laps around an Olympic-sized track.
  Fact 1.0: The race consists of 25 laps.
  Fact 1.1: The 25 laps are around an Olympic-sized track.

Retrieved items:
<item 1>
Harry Prowell
Harry Prowell A.A.(10 July 1936 – 27 June 2000) was a Guyanese long distance runner who represented Guyana in the Marathon at the 1968 Summer Olympics in Mexico
City, Mexico.

 He is known to be one of the greatest Marathon runners Guyana has ever produced, setting the national record in 1968.

 To date, he is the only Guyanese ever to compete in the Marathon at the Summer Olympic Games and one of the most prominent Indo-Caribbean long distance runners
of his time.

 He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth
Games in Edinburgh, Scotland.
</item 1>

<item 2>
The European 10,000m Cup, organized annually by European Athletics since 1997, grew from the Iberian championships and was renamed in 2005. Track 10,000m races
cover 25 laps; the 1500m is a major Olympic middle-distance event. Charlottesville’s annual Bill Steers Men’s 4-Miler, held since 2004, promotes men’s health.
European under-20 championships evolved from junior games (1964, 1968)
</item 2>

<item 3>
Mexican runner Nora Rocha won the 10,000m gold at the 1999 Pan American Games. Paraguayan tennis player Rossana de los Ríos reached No. 51 in singles, won the
1992 French Open junior title, and represented Paraguay at two Olympics. Argentine Paola Suárez won eight Grand Slam doubles titles with Virginia Ruano Pascual
and reached the 2004 French Open singles semifinals; in 1999 she won the Brasil
</item 3>

<item 4>
10,000 metres
The 10,000 metres or 10,000-meter run is a common long-distance track running event.

 The event is part of the athletics programme at the Olympic Games and the World Championships in Athletics and is common at championship level events.

 The race consists of 25 laps around an Olympic-sized track.

 It is less commonly held at track and field meetings, due to its duration.

 The 10,000 metres track race is usually distinguished from its road running counterpart, the 10K run, by its reference to the distance in metres rather than
kilometres.
</item 4>

<item 5>
Dave Bailey (athlete)
David Bailey (born March 17, 1945 in Toronto, Ontario) is a retired track and field athlete, who represented Canada at the 1968 Summer Olympics in the men's
1.500 metres.

 He was the first Canadian to run the mile in less than 4 minutes (3:59.1) in San Diego, CA on June 11, 1966 and the first Canadian to the run the mile in less
than 4 minutes in Canada (3:57.7)in Toronto on July 22, 1967.

 A resident of Willowdale, Ontario he won the bronze medal in this event at the 1967 Pan American Games and the silver medal in this event at the 1968 World
University Games He was two-time Canadian Universities Track and Field Athlete of Year (1965, 1967) and two-time inductee into the University of Toronto Sports
Hall of Fame (individually 1998 and team 2003).
</item 5>

<item 6>
The 2011 Pan American Games in Guadalajara involved about 6,000 athletes from 42 nations; Toronto hosted the 2015 edition across 18 Golden Horseshoe
communities. Port Moresby’s 2015 Pacific Games featured Kiribati, Australia’s debut, and events including basketball, football and powerlifting. Papua New
Guinea dominated women’s powerlifting; Nauru led the men’s, while Tuvalu earned its first Pacific Games gold. Vanuatu
</item 6>

<item 7>
Canadian athletes Ross MacKenzie and Brian MacLaren earned 1966 Commonwealth and 1967 Pan American relay silver; both competed in Olympic 400m. Guyanese Harry
Prowell set a marathon record and remains Guyana’s sole Olympic marathoner. Canadian miler Dave Bailey broke four minutes and won Pan Am bronze; Siegmar
Ohlemann won Pan Am 800m silver. Geri Donnelly and China’s Liu Ailing were World Cup footballers; Liu also
</item 7>

<item 8>
Guadalajara’s 2011 Pan American Games hosted 6,000 athletes from 42 nations; Toronto’s 2015 Games spanned 18 communities. Port Moresby’s 2015 Pacific Games
included Australia’s debut: Papua New Guinea led women’s powerlifting, Nauru men’s, and Tuvalu won its first gold; Nicolas N’Godrela earned three tennis golds.
Profiles span footballers, champion jockeys (Tony McCoy: 4
</item 8>

<item 9>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 9>

<item 10>
Brian MacLaren
Brian MacLaren (born 21 December 1943) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1972 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Bill Crothers, Ross MacKenzie, and Robert McLaren) and third in the 1967 Pan
American Games 800 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Ross MacKenzie and Bill Crothers.
</item 10>
=== END ITEM id=bc1e6feb28443b84 ===

=== ITEM id=77f841ebef4a247f ===
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
Nearer the Moon
Nearer the Moon: From a Journal of Love (full title "Nearer the Moon: From a Journal of Love, the Unexpurgated Diary of Anaïs Nin (1937–1939)") is a 1996 book
based upon material excerpted from the unpublished diaries of Anaïs Nin.

 It corresponds temporally to part of Nin's published diaries.

 It consists mainly of material that was left out of the published version because it would have hurt people involved or their relationships with Anaïs Nin had
it been published at the time.
</item 2>

<item 3>
Nobel Prize in Literature
Since 1901, the Nobel Prize in Literature (Swedish: "Nobelpriset i litteratur" ) has been awarded annually to an author from any country who has, in the words
of the will of Alfred Nobel, produced "in the field of literature the most outstanding work in an ideal direction" (original Swedish: "den som inom litteraturen
har producerat det mest framstående verket i en idealisk riktning").

 Though individual works are sometimes cited as being particularly noteworthy, here "work" refers to an author's work as a whole.

 The Swedish Academy decides who, if anyone, will receive the prize in any given year.

 The academy announces the name of the chosen laureate in early October.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895; the others are the Nobel Prize in Chemistry, Nobel Prize in Physics, Nobel
Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 3>

<item 4>
Seduction of the Minotaur
Seduction of the Minotaur is an autobiographical novel by the mixed nationality writer Anaïs Nin, the last part of her "Cities of the Interior" sequence.

 It is about a woman named Lillian, and her self-psychoanalysis.

 The setting is taken from Anaïs' diary account of her first trip to Acapulco in 1947, and the novel repeats much of the first part of "The Diary of Anaïs Nin"
volume V.

 Since the author was concerned with psychology rather than physical adventure, there is actually less violence in the novel than in the diary account.

 The exception is that the doctor allows himself to be shot because he is loved only as a doctor and never as a man, perhaps patterned after her understanding
of Otto Rank's death.
</item 4>

<item 5>
Graph query row:
person: Odysseas Elytis
award: Nobel Prize in Literature
</item 5>

<item 6>
The Diary of Anaïs Nin
The Diary of Anaïs Nin is the published version of Anaïs Nin's own private manuscript diary, which she began at age 11 in 1914 during a trip from Europe to New
York with her mother and two brothers.

 Anaïs Nin would later say she had begun the diary as a letter to her father, Cuban composer Joaquín Nin, who had abandoned the family a few years earlier.
</item 6>

<item 7>
Anaïs Nin
Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban
parents in France, where she was also raised.

 She spent some time in Spain and Cuba, but lived most of her life in the United States, where she became an established author.

 She wrote journals (which span more than 60 years, beginning when she was 11 years old and ending shortly before her death), novels, critical studies, essays,
short stories, and erotica.

 A great deal of her work, including "Delta of Venus" and "Little Birds", was published posthumously.
</item 7>

<item 8>
Collages (novel)
Collages, published in 1964, was Anaïs Nin's last published novel (excluding her erotica).

 It is very different from the previous novels of the "Cities of the Interior" series, because it contains none of the familiar characters in those novels.

 It is also different in that it takes place on two continents, has about two dozen important characters, and for the first time in Anaïs Nin's work the female
lead is not seeking psychological wholeness.

 Most of Anaïs Nin's other novels only have a few main characters, there is rarely much geographical movement in the other novels, and usually the lead
character is a woman seeking psychological wholeness.
</item 8>

<item 9>
Odysseas Elytis
Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as
a major exponent of romantic modernism in Greece and the world.

 In 1979 he was awarded the Nobel Prize in Literature.
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
=== END ITEM id=77f841ebef4a247f ===

=== ITEM id=aa73a888e26f2881 ===
Gold sentences and facts:
Sentence 0:  The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
  Fact 0.0: She was the daughter of the American Consul in London.
  Fact 0.1: She was the first First Lady to be born outside the United States.
  Fact 0.2: She was the first First Lady to be born outside areas that were later to become part of the United States.
  Fact 0.3: Such areas include the Thirteen Colonies.
  Fact 0.4: The distinction of being the first First Lady born outside these areas was not replicated for 192 years.
  Fact 0.5: Melania Trump replicated this distinction 192 years later.
Sentence 1:  Before marrying future President Donald Trump in 2005, she worked as a fashion model; by 2016 she considered herself a "full-time mom."
  Fact 1.0: Before marrying future President Donald Trump in 2005, she worked as a fashion model.
  Fact 1.1: By 2016, she considered herself a 'full-time mom.'

Retrieved items:
<item 1>
Office of the First Lady of the United States
The Office of the First Lady of the United States is the staff accountable to the First Lady of the United States.

 The office and its responsibilities, while not mandated, have grown as the role of the First Lady has grown and formalized through the history of the United
States.

 The Office of the First Lady is an entity of the White House Office, part of the Executive Office of the President.
</item 1>

<item 2>
Vicki Risch
Vicki Risch is the former First Lady of Idaho and the wife of U.S. Sen. Jim Risch, who served as Governor of Idaho in 2006.

 She became First Lady on May 26, 2006, when her husband succeeded former Gov. Dirk Kempthorne, who resigned to become United States Secretary of the Interior.

 Mrs. Risch succeeded former First Lady Patricia Kempthorne who had held the post for over seven years.

 Mrs. Risch served as First Lady until January 2007, as her husband did not seek a full term as governor, but rather was reelected to his old post as lieutenant
governor.
</item 2>

<item 3>
Abigail Adams
Abigail Adams ("née" Smith; November 22  1744 – October 28, 1818) was the closest advisor and wife of John Adams, as well as the mother of John Quincy Adams.

 She is sometimes considered to have been a Founder of the United States, and is now designated as the first Second Lady and second First Lady of the United
States, although these titles were not in use at the time.
</item 3>

<item 4>
Mary Elizabeth Bliss
Mary Elizabeth "Betty" Taylor Bliss Dandridge, born Mary Elizabeth Taylor (April 20, 1824 – July 25, 1909), was the youngest of the three surviving daughters of
President Zachary Taylor (1849-1850) and Margaret Mackall Smith Taylor.
</item 4>

<item 5>
Doreen Ketchens
Doreen Ketchens (born October 3, 1966 in New Orleans, Louisiana) is a jazz clarinetist, who performs Dixieland and Trad Jazz.

 She is one of the first and few female bandleaders in New Orleans, and a musical educator.

 She has performed at concert halls and music festivals, at U.S. Embassies and decades of weekly performances in Dixieland's tradition in the Royal Street
Performing Arts Zone in the French Quarter of New Orleans with her band, Doreen's Jazz New Orleans.

 Ketchens has performed for four U.S. Presidents: Bill Clinton, George Bush Sr., Ronald Reagan and Jimmy Carter, and is widely considered one of the cultural
ambassadors of New Orleans and of the traditional music
</item 5>

<item 6>
Melania Trump
Melania Trump ( ; born Melanija Knavs; ] , Germanized to Melania Knauss; April 26, 1970) is the current First Lady of the United States.

 Before marrying future President Donald Trump in 2005, she worked as a fashion model; by 2016 she considered herself a "full-time mom."
</item 6>

<item 7>
Anna Harrison
Anna Tuthill Symmes Harrison (July 25, 1775 – February 25, 1864), wife of President William Henry Harrison and grandmother of President Benjamin Harrison, was
nominally First Lady of the United States during her husband's one-month term in 1841, but she never entered the White House.

 At the age of 65 years during her husband's presidential term, she is the oldest woman ever to become First Lady, as well as having the distinction of holding
the title for the shortest length of time, and the first person to be widowed while holding the title.

 She was the last First Lady to have been born in British America.
</item 7>

<item 8>
List of First Ladies of the United States
The First Lady of the United States is the hostess of the White House.

 The position is traditionally filled by the wife of the President of the United States, but, on occasion, the title has been applied to women who were not
presidents’ wives, such as when the president was a bachelor or widower, or when the wife of the president was unable to fulfill the duties of the First Lady
herself.

 The First Lady is not an elected position; it carries no official duties and receives no salary.

 Nonetheless, she attends many official ceremonies and functions of state either along with or in place of the president.

 Traditionally, the First Lady does not hold outside employment while occupying the office.

 She has her own staff, including the White House Social Secretary, the Chief of Staff, the Press Secretary, the Chief Floral Designer, and the Executive Chef.

 The Office of the First Lady is also in charge of all social and ceremonial events of the White House, and is a branch of the Executive Office of the
President.
</item 8>

<item 9>
Louisa Adams
Louisa Catherine Johnson Adams (February 12, 1775 – May 15, 1852), wife of John Quincy Adams, was First Lady of the United States from 1825 to 1829.

 The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become
part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
</item 9>

<item 10>
List of current United States first spouses
In the United States, the first spouse (first lady for women, first gentleman for men) is the term used to refer to the spouse of a chief executive - that is,
of the spouse of the President of the United States (the First Lady of the United States and the First Gentleman of the United States) and the spouses of the
governors of the 50 U.S. states and U.S. territories (Guam, Northern Mariana Islands, Puerto Rico, American Samoa, and the United States Virgin Islands) and the
spouse of the mayor of the District of Columbia.

 (The spouses of many mayors are often called the "first lady" or "first gentleman" of the city as well, and the use of the terms sometimes extends even to the
spouses of college presidents).
</item 10>
=== END ITEM id=aa73a888e26f2881 ===

=== ITEM id=56d43be1c45a712d ===
Gold sentences and facts:
Sentence 0:  In return, William made Copsi earl of Northumbria and sent him back to York.
  Fact 0.0: William made Copsi earl of Northumbria in return.
  Fact 0.1: William sent Copsi back to York.
Sentence 1: York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.
  Fact 1.0: York is a historic walled city.
  Fact 1.1: York is at the confluence of the rivers Ouse and Foss.
  Fact 1.2: York is in North Yorkshire, England.

Retrieved items:
<item 1>
Earl Cowper
He represented Hertford in Parliament.

 He was succeeded by his eldest son, the aforementioned William Cowper, the third Baronet, who was elevated to the peerage as Baron Cowper in 1706 and made Earl
Cowper in 1718.

 In 1706 Lord Cowper married as his second wife Mary Clavering, daughter of John Clavering, of Chopwell, County Durham.
</item 1>

<item 2>
St Chad's College, Durham
St Chad's College is a recognised (independent) college of Durham University in England, founded in 1904 as an Anglican hall for the training of Church of
England clergy.

 The main part of the college is located on the Bailey, occupying nine historic buildings at the east end of Durham Cathedral.

 It neighbours Hatfield College to its north, while St John's College and St Cuthbert's Society are to its south.

 The college is named after St Chad of Mercia, a 7th-century bishop.
</item 2>

<item 3>
Earl of Norbury
Earl of Norbury, in the County of Tipperary, is a title in the Peerage of Ireland.

 It was created in 1827, along with the title Viscount Glandine, of Glandine in the King's County, for the Irish politician and judge John Toler, 1st Baron
Norbury, upon his retirement as Chief Justice of the Common Pleas in Ireland.

 The titles were created with special remainder to his second son, Hector, as his eldest son, Daniel, was then considered mentally unwell.

 Lord Norbury had already been created Baron Norbury, of Ballycrenode in the County of Tipperary, in the Peerage of Ireland in 1800, with remainder to the heirs
male of his body.

 Moreover, his wife, Grace Toler (née Graham), had been created Baroness Norwood, of Knockalton in the County of Tipperary, in the Peerage of Ireland in 1797,
with remainder to the heirs male of her body.

 By the time Lord Norbury was raised to the Earldom, his wife had died and their eldest son had succeeded her as 2nd Baron Norwood.
</item 3>

<item 4>
Tripartite Indenture
The Tripartite Indenture was an agreement made in February 1405 between Owain Glyndŵr, Edmund Mortimer, and Henry Percy, 1st Earl of Northumberland, agreeing to
divide England and Wales up between them at the expense of Henry IV.

 Glyndŵr was to be given Wales, and a substantial part of the west of England, including the English portions of the Welsh Marches.

 Northumberland was to have received the north, as well as Northamptonshire, Norfolk, Warwickshire, and Leicestershire.

 The Mortimers were to have received the rest of southern England.
</item 4>

<item 5>
Two Rivers Press
Two Rivers Press is an independent publishing house, based in the English town of Reading.

 Two Rivers Press was founded in 1994 by Peter Hay (1951–2003), a local artist.

 Its name reflects his enthusiasm for the town and its two rivers, the Kennet and the Thames, and its intention to explore "the place where art and history
meet".

 The name also gives a clue to the origins of the company in the, ultimately successful, opposition to Reading's proposed "Cross-Town Route", a road scheme that
would have seriously impacted the point at which the two rivers meet.
</item 5>

<item 6>
Earl Cowper
Earl Cowper (pronounced "Cooper") was a title in the Peerage of Great Britain.

 It was created in 1718 by George I for William Cowper, 1st Baron Cowper, his first Lord Chancellor, with remainder in default of male issue of his own to his
younger brother, Spencer Cowper.

 Cowper had already been created Baron Cowper of Wingham in the County of Kent, in the Peerage of England on 14 December 1706, with normal remainder to the
heirs male of his body, and was made Viscount Fordwich, in the County of Kent, at the same time as he was given the earldom, also Peerage of Great Britain and
with similar remainder.

 He was the great-grandson of William Cowper, who was created a Baronet, of Ratling Court in the County of Kent, in the Baronetage of England on 4 March 1642.

 The latter was succeeded by his grandson, the second Baronet.

 He represented Hertford in Parliament.
</item 6>

<item 7>
Copsi
Copsi (or Copsig; Old English: "Cōpsige" ) [died 1067] was a Northumbrian magnate in late Anglo-Saxon England.

 He was a supporter of Tostig, and was exiled along with him in 1065.

 Copsi soon fled to Orkney (then a part of Norway).

 The next year (1066), he joined Tostig at Sandwich, in Kent, with 17 ships.

 Copsi survived Tostig's defeat at Stamford Bridge, and when William the Conqueror prevailed at Hastings he travelled, in March 1067, to pay William homage at
Barking (where William was staying while his tower was being constructed in London).

 In return, William made Copsi earl of Northumbria and sent him back to York.

 Copsi's rule lasted a mere five weeks, at which time he was murdered by Osulf, son of Eadulf III of the ancient Bernician family which had historically
governed the area from Bamburgh, at Newburn-upon-Tyne.
</item 7>

<item 8>
Copsi
Osulf, however, only ruled as earl until that autumn, when he was killed by an outlaw he was tracking.
</item 8>

<item 9>
Two Rivers (Red River of the North)
The Two Rivers is a 7.0 mi river in Kittson County, northwestern Minnesota, in the United States.

 Formed by the North Branch of the Two Rivers and the South Branch of the Two Rivers, it is a tributary of the Red River of the North, with its outflow
traveling north through Lake Winnipeg and the Nelson River to Hudson Bay.
</item 9>

<item 10>
York
York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.

 The municipality is the traditional county town of the historic county of Yorkshire to which it gives its name.

 The city has a rich heritage and has provided the backdrop to major political events in England throughout much of its two millennia of existence.

 The city offers a wealth of historic attractions, of which York Minster is the most prominent, and a variety of cultural and sporting activities making it a
popular tourist destination for millions.
</item 10>
=== END ITEM id=56d43be1c45a712d ===

=== ITEM id=144fcdd6988ec2a6 ===
Gold sentences and facts:
Sentence 0: Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central business district of Brainerd, a city in Crow Wing County, Minnesota, United States.
  Fact 0.0: Brainerd Lakes Regional Airport has IATA code BRD.
  Fact 0.1: Brainerd Lakes Regional Airport has ICAO code KBRD.
  Fact 0.2: Brainerd Lakes Regional Airport has FAA LID BRD.
  Fact 0.3: Brainerd Lakes Regional Airport is a public use airport.
  Fact 0.4: Brainerd Lakes Regional Airport is located three nautical miles (6 km) northeast of the central business district of Brainerd.
  Fact 0.5: Brainerd is a city in Crow Wing County, Minnesota, United States.
Sentence 1: Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City, Utah in the United States.
  Fact 1.0: Salt Lake City International Airport has IATA code SLC.
  Fact 1.1: Salt Lake City International Airport has ICAO code KSLC.
  Fact 1.2: Salt Lake City International Airport has FAA LID SLC.
  Fact 1.3: Salt Lake City International Airport is a civil-military airport.
  Fact 1.4: Salt Lake City International Airport is located about 4 miles west of Downtown Salt Lake City.
  Fact 1.5: Downtown Salt Lake City is in Utah, United States.

Retrieved items:
<item 1>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 1>

<item 2>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 2>

<item 3>
Arena (UTA station)
Arena is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit Authority's
(UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line has service from the Salt Lake City International Airport and to West Valley City via Downtown Salt Lake City.
</item 3>

<item 4>
Salt Lake City Air Route Traffic Control Center
Salt Lake City Air Route Traffic Control Center (ZLC) (radio communications "Salt Lake Center") is one of 22 FAA Area Control Centers in the United States.

 It is located in Salt Lake City, Utah, adjacent to Salt Lake City International Airport.

 It was opened in 1939 and was originally located on the third floor of the old Salt Lake City International Airport terminal.

 The Salt Lake Center (ZLC) covers one of the largest geographical areas of any other control center, totaling approximately 350,000 squares miles.
</item 4>

<item 5>
Temple Square (UTA station)
Temple Square is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit
Authority's (UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 5>

<item 6>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 6>

<item 7>
North Temple Bridge/Guadalupe (UTA station)
North Temple Bridge/Guadalupe is a light rail and commuter rail station in Salt Lake City, Utah, United States served by the Green Line of Utah Transit
Authority's (UTA) TRAX light rail system and by the "FrontRunner", UTA's commuter rail train that operates along the Wasatch Front with service from Pleasant
View in northern Weber County through Ogden, Davis County, Salt Lake City, and Salt Lake County to Provo in central Utah County.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City) and connects with the rest of
the TRAX system and UTA's S Line streetcar.

 Although the official name of the station is "North Temple Bridge/Guadalupe", it often referenced by UTA as simply North Temple.
</item 7>

<item 8>
City Center (UTA station)
City Center is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit
Authority's (UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 8>

<item 9>
Courthouse (UTA station)
Courthouse is a light rail station in Downtown Salt Lake City, Utah, United States serviced by all three lines of Utah Transit Authority's TRAX light rail
system.

 The Blue Line provides service from Downtown Salt Lake City to Draper.

 The Red Line provides service from the University of Utah to the Daybreak Community of South Jordan.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 9>

<item 10>
900 South (UTA station)
900 South is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by all three lines of Utah Transit Authority's TRAX light rail
system.

 The Blue Line provides service from Downtown Salt Lake City to Draper.

 The Red Line provides service from the University of Utah to the Daybreak Community of South Jordan.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).

 The station opened on 19 September 2005 and is operated by the Utah Transit Authority.

 900 South is notable for being the first infill station constructed along an existing line of the TRAX system.
</item 10>
=== END ITEM id=144fcdd6988ec2a6 ===

=== ITEM id=44fc95850348e5e3 ===
Gold sentences and facts:
Sentence 0: James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more than 70 productions to his credit.
  Fact 0.0: James David Sharman was born on 12 March 1945.
  Fact 0.1: James David Sharman is the son of boxing tent entrepreneur Jimmy Sharman.
  Fact 0.2: James David Sharman is an Australian director.
  Fact 0.3: James David Sharman is an Australian writer for film and stage.
  Fact 0.4: James David Sharman has more than 70 productions to his credit.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born on August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is an American director.
  Fact 1.2: Ava Marie DuVernay is an American screenwriter.
  Fact 1.3: Ava Marie DuVernay is an American film marketer.
  Fact 1.4: Ava Marie DuVernay is an American film distributor.

Retrieved items:
<item 1>
Jim Sharman
James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more
than 70 productions to his credit.

 He is renowned in Australia for his work as a theatre director from the 1960s to the present, and is best known internationally as the director of the 1973
theatrical hit "The Rocky Horror Show", its film adaptation "The Rocky Horror Picture Show" (1975) and the film's follow-up "Shock Treatment" (1981).
</item 1>

<item 2>
Lowell Sherman
Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.

 At a time when it was highly unusual, he was both the actor and director on several films in the early 1930s, before completely transitioning to the role of
director.

 At the height of his career, after scoring huge successes with his direction of the films "She Done Him Wrong" and "Morning Glory" (which starred Mae West and
won her first Academy Award for Katharine Hepburn respectively), he succumbed to pneumonia after a brief illness.
</item 2>

<item 3>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 3>

<item 4>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 4>

<item 5>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 5>

<item 6>
List of accolades received by Selma (film)
"Selma" is a 2014 American historical drama film directed by Ava DuVernay, and produced by Dede Gardner, Jeremy Kleiner, Christian Colson, and Oprah Winfrey.

 The screenplay was written by Paul Webb.

 The film follows the events leading up to and during the Selma to Montgomery marches, and the resulting establishment of the Voting Rights Act of 1965 which
prohibited racial discrimination in voting in the United States.

 The film stars David Oyelowo as civil rights activist Martin Luther King Jr., Tom Wilkinson as President Lyndon B. Johnson, and Tim Roth as Governor of Alabama
George Wallace.

 Carmen Ejogo, Dylan Baker, Wendell Pierce, Common, and Winfrey feature in supporting roles.
</item 6>

<item 7>
Graph query row:
entity: Ava DuVernay
occupation: ['director', 'screenwriter', 'film marketer', 'film distributor']
relationship: CREATED_BY
role: director
related: 13th
</item 7>

<item 8>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 8>

<item 9>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 9>

<item 10>
Graph query row:
entity: Ava DuVernay
occupation: ['director', 'screenwriter', 'film marketer', 'film distributor']
relationship: CREATED_BY
role: None
related: Selma
</item 10>
=== END ITEM id=44fc95850348e5e3 ===

=== ITEM id=76a0dedadd9b4afc ===
Gold sentences and facts:
Sentence 0: Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.
  Fact 0.0: Wenling is a coastal county-level city.
  Fact 0.1: Wenling is in the municipal region of Taizhou.
  Fact 0.2: Taizhou is in southeastern Zhejiang province.
  Fact 0.3: Wenling is in China.
  Fact 0.4: In the Wenling dialect, Wenling is called Uen-lin Zy.
Sentence 1: Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.
  Fact 1.0: Ruili is a county-level city of Dehong Prefecture.
  Fact 1.1: Ruili is in the west of Yunnan province.
  Fact 1.2: Yunnan province is in the People's Republic of China.
  Fact 1.3: Ruili's Burmese name is ရွှေလီ.
  Fact 1.4: Ruili's Thai name is เมืองมาว.

Retrieved items:
<item 1>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 1>

<item 2>
Linzhou, Henan
Linzhou (), formerly Lin County (林县), is a county-level city in Anyang, Henan, China.

 Adjacent to Shanxi Province and Hebei Province, it is located in the north of Henan Province and at the eastern foot of the Taihang Mountains.

 It covers an area of 2046 square kilometers and has a population of about one million.
</item 2>

<item 3>
Dali–Ruili Railway
The Dali–Ruili Railway or Darui Railway (), is a single-track electrified railroad under construction in Yunnan Province of Southwest China.

 The line is slated to run 336.39 km from Dali in central Yunnan to Ruili in southwestern Yunnan on the border with Myanmar.

 The line traverses rugged terrain, and bridges and tunnels will account for 75% of the total track length, including a 36-km tunnel through the Gaoligong
Mountains.

 Construction began in May 2011 and is scheduled to take six years.

 Cities and towns along route include Dali, Yangbi Yi Autonomous County, Yongping County, Baoshan, Mang City (also known as Mangshi or Dehong) and Ruili.
</item 3>

<item 4>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 4>

<item 5>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 5>

<item 6>
Yandang Mountains
Yandang Mountains or Yandangshan (Chinese:  雁蕩山 ,  雁荡山 ,  "Yàndàng Shān",  "Wild Goose Pond Mountain(s)") refers, in the broad sense, to a coastal mountain
range in southeastern Zhejiang province in eastern China, covering much of the prefecture-level city of Wenzhou (from Pingyang County in the south to Yueqing
County in the northeast) and extending to the county-level city of Wenling in Taizhou prefecture.

 The mountain range is divided in two by the Oujiang River, the two parts being the North Yandang and South Yandang.

 More narrowly, Yandangshan is also used more narrowly to refer to Mount Yandang , a specific part of the North Yandang around an ancient caldera near a small
town of the same name (雁荡镇 , "Yàndàng Zhèn").

 The highest peaks of North Yandang are located here, and this is also the main tourist spot.
</item 6>

<item 7>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 7>

<item 8>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 8>

<item 9>
Ruili
Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.

 It is a major border crossing between China and Myanmar, with the town of Muse located across the border.
</item 9>

<item 10>
Ruili Border Economic Cooperation Zone
Ruili Border Economic Cooperation Zone (RLBECZ) is a Chinese State Council-approved Industrial Park based in Ruili City, Dehong Prefecture, Yunnan, China,
founded in 1992 and was established to promote trade between China and Myanmar.

 The zone connects Ruili with Man-wing City.
</item 10>
=== END ITEM id=76a0dedadd9b4afc ===

=== ITEM id=3156982160a1ab40 ===
Gold sentences and facts:
Sentence 0:  The station is owned by Gray Television, as part of a duopoly with CBS affiliate KWCH-DT (channel 12); Gray also operates Univision affiliate KDCU-DT (channel 31) under a joint sales agreement with owner Entravision Communications Corporation.
  Fact 0.0: The station is owned by Gray Television.
  Fact 0.1: The station is part of a duopoly with CBS affiliate KWCH-DT (channel 12).
  Fact 0.2: Gray Television also operates Univision affiliate KDCU-DT (channel 31).
  Fact 0.3: Gray Television operates KDCU-DT under a joint sales agreement with its owner, Entravision Communications Corporation.
Sentence 1: KWCH-DT, virtual channel 12 (UHF digital channel 19), is a CBS-affiliated television station serving Wichita, Kansas, United States that is licensed to Hutchinson.
  Fact 1.0: KWCH-DT is virtual channel 12.
  Fact 1.1: KWCH-DT broadcasts on UHF digital channel 19.
  Fact 1.2: KWCH-DT is a CBS-affiliated television station.
  Fact 1.3: KWCH-DT serves Wichita, Kansas, United States.
  Fact 1.4: KWCH-DT is licensed to Hutchinson.

Retrieved items:
<item 1>
KSCW-DT
KSCW-DT, virtual channel 33 (VHF digital channel 12), is a CW-affiliated television station licensed to Wichita, Kansas, United States.

 The station is owned by Gray Television, as part of a duopoly with CBS affiliate KWCH-DT (channel 12); Gray also operates Univision affiliate KDCU-DT (channel
31) under a joint sales agreement with owner Entravision Communications Corporation.

 KSCW and KWCH share studio facilities and KDCU's master control operations are located on East 37th Street North in northeastern Wichita; KSCW maintains
transmitter facilities located in rural northeastern Reno County (due south of Buhler).
</item 1>

<item 2>
WDJT-TV
WDJT-TV, virtual channel 58 (UHF digital channel 46), is a CBS-affiliated television station licensed to Milwaukee, Wisconsin, United States.

 The station is owned by Chicago-based Weigel Broadcasting, as part of a duopoly with Racine-licensed independent station WMLW-TV (channel 49), and is also a
sister station to MeTV owned-and-operated station WBME-CD (channel 41, which is also carried via WDJT's DT2 subchannel) and Telemundo affiliate WYTU-LD (channel
63).

 All four stations share studio facilities located on South 60th Street in Milwaukee (near West Allis); WDJT's transmitter is located in Milwaukee's Lincoln
Park (next to the transmitter belonging to ABC affiliate WISN-TV, channel 12).
</item 2>

<item 3>
KDCU-DT
KDCU-DT, virtual and UHF digital channel 31, is a Univision-affiliated television station serving Wichita, Kansas, United States that is licensed to Derby.

 The station is owned by the Entravision Communications Corporation; Gray Television, which owns CBS affiliate KWCH-DT (channel 12) and CW affiliate KSCW-DT
(channel 33), operates KDCU under a joint sales agreement.

 KDCU's offices and master control operations are based at KWCH and KSCW's joint studio facilities on 37th Street in northeast Wichita; KDCU maintains
transmitter facilities located in rural northwestern Sedgwick County (northeast of Colwich).

 On cable, the station is available in standard definition on Cox Communications channel 15 and AT&T U-verse channel 31, and in high definition on Cox digital
channel 2015 and U-verse channel 1031.
</item 3>

<item 4>
WTLV
WTLV, virtual channel 12 (VHF digital channel 13), is an NBC-affiliated television station licensed to Jacksonville, Florida, United States.

 The station is owned by the Tegna Media division of Tegna, Inc., as part of a duopoly with Orange Park-licensed ABC affiliate WJXX (channel 25).

 The two stations share studio facilities on East Adams Street (near EverBank Field) in downtown Jacksonville; WTLV's transmitter is located on Eve Drive in the
city's Killarney Shores section.
</item 4>

<item 5>
KMBC-TV
KMBC-TV, virtual channel 9 (UHF digital channel 29), is the ABC-affiliated television station licensed to Kansas City, Missouri, United States and also serving
Kansas City, Kansas.

 The station is owned by the Hearst Television subsidiary of the Hearst Corporation, as part of a duopoly with CW affiliate KCWE (channel 29).

 The two stations share studio facilities located at the Winchester Business Center on Winchester Avenue and East 63rd Street (near Swope Park, off of I-435) in
southeastern Kansas City, Missouri; KMBC-TV maintains transmitter facilities located near the Blue River in eastern Kansas City.

 On cable, KMBC-TV is available on Charter Spectrum, Comcast Xfinity and Consolidated Communications channel 12, and Google Fiber and AT&T U-verse channel 9.

 There is a high definition feed provided on Spectrum digital channel 1200, Xfinity channel 812, Consolidated channel 610 and U-verse channel 1009.
</item 5>

<item 6>
WJXX
WJXX, virtual channel 25 (VHF digital channel 10), is an ABC-affiliated television station serving Jacksonville, Florida, United States that is licensed to
Orange Park.

 The station is owned by the Tegna Media division of Tegna, Inc., as part of a duopoly with Jacksonville-licensed NBC affiliate WTLV (channel 12) (ironically a
former ABC affiliate itself from 1980 to 1988).

 The two stations share studio facilities on East Adams Street (near EverBank Field) in downtown Jacksonville; WJXX's transmitter is located on Eve Drive in the
city's Kilarney Shores section.
</item 6>

<item 7>
KXMB-TV
KXMB-TV, channel 12, is the CBS-affiliated television station for Bismarck, North Dakota.

 It broadcasts a high definition digital signal on VHF channel 12 (or virtual channel 12.1 via PSIP) from a transmitter near St. Anthony.

 The station can also be seen on Midcontinent cable channel 11 in Bismarck-Mandan, and cable channel 12 in most other areas.

 There is a high definition feed provided on Midcontinent digital channel 611.
</item 7>

<item 8>
WFLD
WFLD, virtual channel 32 (UHF digital channel 31), is a Fox owned-and-operated television station licensed to Chicago, Illinois, United States.

 The station is owned by the Fox Television Stations division of 21st Century Fox, and operates as part of a television duopoly with Gary, Indiana-licensed
primary CW affiliate and secondary MyNetworkTV owned-and-operated station WPWR-TV (channel 50).

 The two stations share studio and office facilities located at Michigan Plaza on North Michigan Avenue in the Chicago Loop; WFLD's transmitter is based at the
Willis Tower on South Wacker Drive in the Loop business district.

 On cable, the station can be seen on Comcast Xfinity channel 12 in most parts of the Chicago area.
</item 8>

<item 9>
KWCH-DT
KWCH-DT, virtual channel 12 (UHF digital channel 19), is a CBS-affiliated television station serving Wichita, Kansas, United States that is licensed to
Hutchinson.

 The station is owned by Gray Television, as part of a duopoly with CW affiliate KSCW-DT (channel 33); Gray also operates Univision affiliate KDCU-DT (channel
31) under a joint sales agreement with owner Entravision Communications Corporation.

 KWCH and KSCW share studio facilities and KDCU's master control operations are located on 37th Street in northeast Wichita; KWCH maintains transmitter
facilities located in rural northeastern Reno County (south-southeast of Buhler).
</item 9>

<item 10>
KHSL-TV
KHSL-TV, virtual channel 12 (UHF digital channel 43) is a CBS-affiliated television station located in Chico, California, United States, that also serves
Redding.

 The station is owned by Heartland Media as part of a duopoly with NBC affiliate KNVN (channel 24), which is owned by Maxair Media but operated by Heartland
through a local marketing agreement.

 The two stations share studios at the corner of Eaton Road and Silverbell Road on the northwest side of Chico, while its transmitter is located along Cohasset
Road in rural Butte County northwest of Paradise.
</item 10>
=== END ITEM id=3156982160a1ab40 ===
