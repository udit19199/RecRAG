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

=== ITEM id=378211e4681a2234 ===
Gold sentences and facts:
Sentence 0: Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.
  Fact 0.0: Wangaratta Wahine is an album by the Australian band The Captain Matchbox Whoopee Band.
  Fact 0.1: Wangaratta Wahine was released in 1974.
  Fact 0.2: Wangaratta Wahine was the second album of The Captain Matchbox Whoopee Band.
Sentence 1:  He was declared an Australian Living Treasure by the National Trust of Australia in 1999.
  Fact 1.0: The subject of this sentence was declared an Australian Living Treasure by the National Trust of Australia.
  Fact 1.1: The subject of this sentence received this declaration in 1999.

Retrieved items:
<item 1>
He Will Have His Way
He Will Have His Way is a compilation album featuring male Australian and New Zealand musicians performing songs written by Neil Finn and Tim Finn (The Finn
Brothers), who are best known as members of Split Enz and Crowded House.

 It was released on 12 November 2010 and is a sequel to "She Will Have Her Way", a 2005 album featuring female Australian and New Zealand musicians.

 The two albums were later released together with extra tracks as the double CD "They Will Have Their Way".
</item 1>

<item 2>
Leah Flanagan
Leah Flanagan is a singer-songwriter from Sydney, NSW.

 Leah has appeared on Australian TV shows Spicks & Specks and RockQuiz, has released 2 albums and toured extensively through Australia with her music and part
of festival ensembles.

 Leah’s status as an Indigenous musician is one of pride and respect; she has helped make history with such recent essential albums as Archie Roach’s 25th
anniversary reissue of Charcoal Lane and Buried Country 1.5, The Story of Aboriginal Country Music.

 Her latest album Saudades was recorded at the studio of Midnight Oil's Jim Moginie and mixed by Paul McKercher (You Am I, Augie March)
</item 2>

<item 3>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 3>

<item 4>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 4>

<item 5>
Wangaratta Wahine
Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.

 It was recorded at top recording studio Armstrong Studios in Melbourne and the cover art was by famed Australian artist Michael Leunig.
</item 5>

<item 6>
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
</item 6>

<item 7>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 7>

<item 8>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 8>

<item 9>
Live at Winterland 1978 (Avengers album)
Live at Winterland 1978 is a live album by the Avengers.

 It was released through online music stores on February 16, 2010.

 The album features a recording of their set on January 14, 1978 at the Winterland Ballroom in San Francisco, CA, when they opened for the Sex Pistols on what
would become the Sex Pistols final show, before reuniting years later.

 The Sex Pistols' set has been released on an album of the same name.

 Songs from this album were previously released on a couple 7" bootlegs titled "Penelope" and "Summer of Hate".
</item 9>

<item 10>
Winwood (album)
Winwood is the first compilation album of music featuring Steve Winwood.

 This two-record set was issued in 1971 by United Artists Records and features music which Winwood performed with The Spencer Davis Group, Powerhouse, Traffic
and Blind Faith.

 UA Records issued this album after Winwood's band Traffic left UA when their home label Island Records set up their own American operation.

 Issued without Winwood's authorization as catalogue number UAS-9950, it was taken off the market after legal action by Winwood and Island Records.

 It was then reissued with minor changes as catalogue number UAS-9964.

 Currently out of print, it was issued on CD by Universal Music of Japan for the Japanese market.
</item 10>
=== END ITEM id=378211e4681a2234 ===

=== ITEM id=c37da47c295d6ffe ===
Gold sentences and facts:
Sentence 0: Eryngium is a genus of flowering plants in the family Apiaceae.
  Fact 0.0: Eryngium is a genus of flowering plants.
  Fact 0.1: Eryngium is in the family Apiaceae.
Sentence 1: Arnebia is a genus of flowering plants in the family Boraginaceae.
  Fact 1.0: Arnebia is a genus of flowering plants.
  Fact 1.1: Arnebia is in the family Boraginaceae.

Retrieved items:
<item 1>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 1>

<item 2>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 2>

<item 3>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 3>

<item 4>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 4>

<item 5>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 5>

<item 6>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 6>

<item 7>
Vascular plant
Vascular plants (from Latin "vasculum": duct), also known as tracheophytes (from the equivalent Greek term "trachea") and also higher plants, form a large group
of plants (c. 308,312 accepted known species ) that are defined as those land plants that have lignified tissues (the xylem) for conducting water and minerals
throughout the plant.

 They also have a specialized non-lignified tissue (the phloem) to conduct products of photosynthesis.

 Vascular plants include the clubmosses, horsetails, ferns, gymnosperms (including conifers) and angiosperms (flowering plants).

 Scientific names for the group include Tracheophyta and Tracheobionta.
</item 7>

<item 8>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 8>

<item 9>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 9>

<item 10>
Cyanobiont
Several cyanobionts involved with fungi and marine organisms also belong to the genera "Richelia, Calothrix, Synechocystis, Aphanocapsa" and "Anabaena", as well
as the species "Oscillatoria spongeliae".

 Although there are many documented symbioses between cyanobacteria and marine organisms, little is known about the nature of many of these symbioses.

 The possibility of discovering more novel symbiotic relationships is apparent from preliminary microscopic observations.
</item 10>
=== END ITEM id=c37da47c295d6ffe ===

=== ITEM id=9c4895d768017e8c ===
Gold sentences and facts:
Sentence 0:  The 381-ton vessel was armed with a 12-pounder 12cwt QF gun, a 20mm Oerlikon cannon, a .303-inch Vickers machine gun, and four Type D depth charges, and was commissioned into the RAN on 11 June 1941.
  Fact 0.0: The vessel weighed 381 tons.
  Fact 0.1: The vessel was armed with a 12-pounder 12cwt QF gun.
  Fact 0.2: The vessel was armed with a 20mm Oerlikon cannon.
  Fact 0.3: The vessel was armed with a .303-inch Vickers machine gun.
  Fact 0.4: The vessel was armed with four Type D depth charges.
  Fact 0.5: The vessel was commissioned into the RAN on 11 June 1941.
Sentence 1:  It was in service from before the First World War until the 1960s, with air-cooled versions of it on many Allied World War I fighter aircraft.
  Fact 1.0: This item was in service from before the First World War until the 1960s.
  Fact 1.1: Air-cooled versions of this item were on many Allied World War I fighter aircraft.

Retrieved items:
<item 1>
HMS Diomede (D92)
HMS "Diomede" was a "Danae"-class cruiser of the Royal Navy.

 Constructed at Vickers Armstrong, Barrow, she was constructed too late to take part in World War I and was consequently completed at the Royal Dockyard,
Portsmouth.

 Between the wars, she served on the China Station, Pacific waters, East Indies Waters and from 1936 onwards, in reserve.

 In World War II she performed four years of arduous war duty, during which time she captured the German blockade runner "Idarwald" .

 Between 22 July 1942 and 24 September 1943 she was converted to a training ship at Rosyth Dockyard.

 In 1945 she was placed in reserve and scrapped a year later.
</item 1>

<item 2>
Goalkeeper CIWS
Goalkeeper is a Dutch close-in weapon system (CIWS) introduced in 1979.

 It is an autonomous and completely automatic weapon system for short-range defence of ships against highly maneuverable missiles, aircraft and fast maneuvering
surface vessels.

 Once activated the system automatically undertakes the entire air defense process from surveillance and detection to destruction, including selection of the
next priority target.
</item 2>

<item 3>
HMS Illustrious (87)
HMS "Illustrious" was the lead ship of her class of aircraft carriers built for the Royal Navy before World War II.

 Her first assignment after completion and working up was with the Mediterranean Fleet, in which her aircraft's most notable achievement was sinking one Italian
battleship and badly damaging two others during the Battle of Taranto in late 1940.

 Two months later the carrier was crippled by German dive bombers and was repaired in the United States.

 After sustaining damage on the voyage home in late 1941 by a collision with her sister ship "Formidable" , "Illustrious" was sent to the Indian Ocean in early
1942 to support the invasion of Vichy French Madagascar (Operation Ironclad).

 After returning home in early 1943, the ship was given a lengthy refit and briefly assigned to the Home Fleet.

 She was transferred to Force H for the Battle of Salerno in mid-1943 and then rejoined the Eastern Fleet in the Indian Ocean at the beginning of 1944.
</item 3>

<item 4>
Weapon system
Weapon System is a United States military term that designated, along with a weapon system number (e.g., WS-110), military experimental (MX) weapons prior to
official naming (e.g., under a military aircraft designation system.

 Preceded by the first Skunk Works program (MX-813 for the Convair XF-92 in 1946), the earliest "WS" designation was the 1954 WS-117L.

 Circa February 1950, an Air Research and Development Command ""study prepared by Maj Gen Gordon P. Saville ... recommended that a 'systems approach' to new
weapons be adopted "[whereby]" development of a weapon "system" required development of support equipment as well as the actual hardware itself.""
</item 4>

<item 5>
AK-630
The AK-630 is a Soviet and Russian fully automatic naval close-in weapon system based on a six-barreled 30 mm rotary cannon.

 In "630", "6" means 6 barrels and "30" means 30 mm.

 It is mounted in an enclosed automatic turret and directed by radar and television detection and tracking.

 The system's primary purpose is defense against anti-ship missiles and other precision guided weapons.

 However it can also be employed against fixed or rotary wing aircraft, ships and other small craft, coastal targets, and floating mines.

 Once operational, this weapon system was rapidly adopted, with up to 8 units installed in every new Soviet warship (from mine-hunters to aircraft carriers),
and hundreds produced in total.
</item 5>

<item 6>
SWARM
The SWARM Remote Weapon System (Stabilised Weapon And Reconnaissance Mount) is a fully armored remote weapon system designed and built by the Thales Group in
Glasgow, Scotland.

 The SWARM system consists of two main assemblies: the Gun Processing and Interface Unit (GPIU), which is operated inside the vehicle, and the external Weapon
and Sensor Platform (WASP).

 It can fire a variety of weapons, and utilize multiple sensors.

 On the US Marine Corps' Gladiator Tactical Unmanned Ground Vehicle (TUGV), it is equipped with a 7.62 mm M240 and day/night sensors.
</item 6>

<item 7>
HMAS Narani
HMAS "Narani" was an auxiliary minesweeper operated by the Royal Australian Navy (RAN) during World War II.

 "Narani" was requisitioned from the Illawarra & South Coast Steam Navigation Company as auxiliaries.

 The 381-ton vessel was armed with a 12-pounder 12cwt QF gun, a 20mm Oerlikon cannon, a .303-inch Vickers machine gun, and four Type D depth charges, and was
commissioned into the RAN on 11 June 1941.
</item 7>

<item 8>
Vickers machine gun
The Vickers machine gun or Vickers gun is a name primarily used to refer to the water-cooled .303 British (7.7 mm) machine gun produced by Vickers Limited,
originally for the British Army.

 The machine gun typically required a six to eight-man team to operate: one fired, one fed the ammunition, the rest helped to carry the weapon, its ammunition,
and spare parts.

 It was in service from before the First World War until the 1960s, with air-cooled versions of it on many Allied World War I fighter aircraft.
</item 8>

<item 9>
Pershing II
The Pershing II Weapon System was a solid-fueled two-stage ballistic missile designed and built by Martin Marietta to replace the Pershing 1a Field Artillery
Missile System as the United States Army's primary nuclear-capable theater-level weapon.

 The U.S. Army replaced the Pershing 1a with the Pershing II Weapon System in 1983 while the German Air Force retained Pershing 1a until all Pershings were
eliminated in 1991.

 The U.S. Army Missile Command (MICOM) managed the development and improvements while the Field Artillery Branch deployed the systems and developed tactical
doctrine.
</item 9>

<item 10>
List of sunken aircraft carriers
With the advent of heavier-than-air flight, the aircraft carrier has become a decisive weapon at sea.

 In 1911 aircraft began to be successfully launched and landed on ships with the successful flight of a Curtiss Pusher aboard the USS "Pennsylvania".

 The British Royal Navy pioneered the first aircraft carrier as floatplanes, as flying boats under performed compared to traditional land based aircraft.

 The first true aircraft carrier was the HMS "Argus" , launched in late 1917 with a complement of 20 aircraft, a flight deck of 550 ft and 68 ft wide.

 The last aircraft carrier sunk in wartime was the Japanese carrier "Amagi", in Kure Harbour in July 1945.

 The greatest loss of life was the 2,046 killed on the "Akitsu Maru"—a converted passenger liner with a small flight deck, carrying the Imperial Japanese Army's
64th Infantry Regiment.
</item 10>
=== END ITEM id=9c4895d768017e8c ===

=== ITEM id=35f226a508789eec ===
Gold sentences and facts:
Sentence 0: Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
  Fact 0.0: Leitneria floridana is also called corkwood.
  Fact 0.1: Leitneria floridana is the sole species in the genus Leitneria.
  Fact 0.2: Leitneria floridana is a deciduous dioecious shrub or small tree.
  Fact 0.3: Leitneria floridana is found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
Sentence 1: Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.
  Fact 1.0: Tricyrtis is a genus of Asian flowering plants.
  Fact 1.1: Tricyrtis is in the lily family.
  Fact 1.2: Tricyrtis has approximately 20 known species.

Retrieved items:
<item 1>
Leitneria
Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United
States states of Arkansas, Florida, Georgia, Missouri and Texas.
</item 1>

<item 2>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 2>

<item 3>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 3>

<item 4>
Tricyrtis
Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.

 The species are commonly known in English as toad lilies.

 The genus has a native range from the Himalayas to eastern Asia, including China, Japan, Philippines and Taiwan, and a few species are cultivated for their
ornamental qualities in other parts of the world.
</item 4>

<item 5>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 5>

<item 6>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 6>

<item 7>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 7>

<item 8>
Traubiinae
Traubiinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 8>

<item 9>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 9>

<item 10>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 10>
=== END ITEM id=35f226a508789eec ===

=== ITEM id=ff16bc2a2af33464 ===
Gold sentences and facts:
Sentence 0: The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.
  Fact 0.0: The soundtrack for the 1968 film "Romeo and Juliet" was composed by Nino Rota.
  Fact 0.1: The soundtrack for the 1968 film "Romeo and Juliet" was conducted by Nino Rota.
Sentence 1: Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
  Fact 1.0: Romeo and Juliet is a 1968 British-Italian romance film.
  Fact 1.1: Romeo and Juliet is based on the tragic play of the same name.
  Fact 1.2: The play Romeo and Juliet was written between 1591 and 1595.
  Fact 1.3: The play Romeo and Juliet was written by William Shakespeare.

Retrieved items:
<item 1>
The Godfather (soundtrack)
The Godfather is the soundtrack from the film of the same name, released in 1972 by Paramount Records, and in 1991 on compact disc by MCA.

 Unless noted, the cues were composed by Nino Rota and conducted by Carlo Savina (who was credited on the LP, but not the CD).

 The song "I Have but One Heart" is sung by Al Martino, who performed it in the film as character Johnny Fontane.
</item 1>

<item 2>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 2>

<item 3>
Amarcord Nino Rota
Amarcord Nino Rota is an album by various artists, recorded as a tribute to composer Nino Rota.
</item 3>

<item 4>
West Side Story (film)
West Side Story is a 1961 American musical romantic drama film directed by Robert Wise and Jerome Robbins.

 The film is an adaptation of the 1957 Broadway musical of the same name, which in turn was inspired by William Shakespeare's play "Romeo and Juliet".

 It stars Natalie Wood, Richard Beymer, Russ Tamblyn, Rita Moreno, and George Chakiris, and was photographed by Daniel L. Fapp, A.S.C., in Super Panavision 70.

 Released on October 18, 1961 through United Artists, the film received high praise from critics and viewers, and became the second highest grossing film of the
year in the United States.

 The film was nominated for 11 Academy Awards and won 10, including Best Picture (as well as a special award for Robbins), becoming the record holder for the
most wins for a movie musical.
</item 4>

<item 5>
Romeo and Juliet (1968 film)
Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
</item 5>

<item 6>
The Cold Nose
The track "We Have to Respect Each Other" samples Shooby Taylor, the human horn, while the opening to "Forty Dollar Rug" samples "Il Teatrino Delle Suore" from
Nino Rota's soundtrack to Giulietta Degli Spiriti.
</item 6>

<item 7>
8½
8½ (Italian title: Otto e mezzo ] ) is a 1963 semi-autobiographical comedy-drama film directed by Federico Fellini.

 Co-scripted by Fellini, Tullio Pinelli, Ennio Flaiano, and Brunello Rondi, it stars Marcello Mastroianni as Guido Anselmi, a famous Italian film director.

 Shot in black-and-white by cinematographer Gianni di Venanzo, the film features a soundtrack by Nino Rota with costume and set designs by Piero Gherardi.
</item 7>

<item 8>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 8>

<item 9>
Il cappello di paglia di Firenze
Il cappello di paglia di Firenze (literally translated as "The Florentine Straw Hat" but usually titled in English language productions as "The Italian Straw
Hat") is an opera by Nino Rota to an Italian-language libretto by the composer and Ernesta Rota, based on the play "Le chapeau de paille d'Italie" by Eugène
Labiche and Marc Michel.
</item 9>

<item 10>
Our Tune
Our Tune is a long-standing feature/segment on British radio presented by broadcaster Simon Bates.

 Having begun by at least 1979 it was originally part of his mid-morning show on BBC Radio 1, where it aired daily throughout the 1980s and early 1990s.

 The feature has more recently been heard on Smooth Radio, where Bates presented the Breakfast Show from 2011 to 2014.

 An edition of "Our Tune" typically features a personal story submitted by a listener together with a song that has significance to the person or situation.

 Many of these stories, which are read out over Nino Rota's Love Theme from "Romeo and Juliet" have a tragic narrative such as illness or death, although not
all end on such an unhappy note.
</item 10>
=== END ITEM id=ff16bc2a2af33464 ===

=== ITEM id=3e07a45e2a1026c0 ===
Gold sentences and facts:
Sentence 0: Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.
  Fact 0.0: Bicycling is a cycling brand.
  Fact 0.1: Bicycling is published by Rodale, Inc. in Emmaus, Pennsylvania.
Sentence 1: National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural affairs.
  Fact 1.0: National Review (NR) is an American magazine.
  Fact 1.1: National Review is semi-monthly.
  Fact 1.2: National Review is conservative.
  Fact 1.3: National Review is an editorial magazine.
  Fact 1.4: National Review focuses on news and commentary pieces on political affairs.
  Fact 1.5: National Review focuses on news and commentary pieces on social affairs.
  Fact 1.6: National Review focuses on news and commentary pieces on cultural affairs.

Retrieved items:
<item 1>
National Review
National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural
affairs.

 The magazine was founded by the author William F. Buckley Jr. in 1955.

 It is currently edited by Rich Lowry.
</item 1>

<item 2>
The Human Life Review
The Human Life Review is a quarterly journal published by the Human Life Foundation since 1975.

 It is devoted to explorations of life issues, primarily abortion, as well as neonaticide, medical genetics, prenatal testing, human cloning, fetal tissue
experimentation, euthanasia and assisted suicide, and also publishes articles dealing with more general questions of family and society.

 It was founded by James Patrick McFadden, formerly associate publisher of "National Review", who had also founded the Human Life Foundation, and is now edited
by his daughter, Maria McFadden.

 It was launched from the offices of "National Review", with the support of William F. Buckley.
</item 2>

<item 3>
Mark Krikorian
Mark Krikorian has been the executive director of the Center for Immigration Studies, a think-tank in Washington, D. C., since 1995.

 The Center describes itself as an "independent, non-partisan research organization" in Washington, D. C., that examines and critiques the impact of immigration
on the United States.

 Animated by a "pro-immigrant, low-immigration vision which seeks fewer immigrants, but a warmer welcome for those admitted", the Center was established in 1985
to provide immigration research.

 Krikorian is a regular contributor to the conservative publication "National Review", and is a regular participant at "National Review Online's" "The Corner."
</item 3>

<item 4>
Kevin D. Williamson
Kevin Daniel Williamson (born September 18, 1972) is the roving correspondent for "National Review."

 He is also the theater critic for "The New Criterion".

 He was previously deputy managing editor at "National Review".
</item 4>

<item 5>
Eliška Drahotová
Eliška Drahotová (] ; born 22 July 1995, Rumburk, Czech Republic) is a Czech athletics competitor in racewalking and road cycling, like her twin sister, Anežka.

 She won the bronze medal in the 2013 European Athletics Junior Championships, while her twin sister Anežka Drahotová took the gold.

 Her global debut in cycling also came that year at the 2013 UCI Road World Championships.

 There she placed 34th in the junior time trial.
</item 5>

<item 6>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 6>

<item 7>
Review
A review is an evaluation of a publication, service, or company such as a movie (a movie review), video game (video game review), musical composition (music
review of a composition or recording), book (book review); a piece of hardware like a car, home appliance, or computer; or an event or performance, such as a
live music concert, play, musical theater show, dance show, or art exhibition.

 In addition to a critical evaluation, the review's author may assign the work a rating to indicate its relative merit.

 More loosely, an author may review current events, trends, or items in the news.

 A compilation of reviews may itself be called a review. "

The New York Review of Books", for instance, is a collection of essays on literature, culture, and current affairs. "

National Review", founded by William F. Buckley, Jr., is an influential conservative magazine, and "Monthly Review" is a long-running socialist periodical.
</item 7>

<item 8>
Charles C. W. Cooke
Charles C. W. Cooke (born November 4, 1984) is the editor of "National Review Online".

 He took the role over after Rich Lowry stepped down in June, 2016 (Lowry remains the editor-in-chief of "National Review").

 Cooke is the author of "The Conservatarian Manifesto" and a frequent guest on HBO's "Real Time with Bill Maher".

 In addition to "National Review", he has written for the "New York Times", the "Washington Post", and the "Los Angeles Times".

 Along with Kevin D. Williamson, he hosts the popular "Mad Dogs and Englishmen" podcast.
</item 8>

<item 9>
Kate O'Beirne
Kate Walsh O'Beirne (September 23, 1949 – April 23, 2017) was the President of National Review Institute.

 She was the Washington editor of "National Review".

 Her column, "Bread and Circuses," covered Congress, politics, and U.S. domestic policy.
</item 9>

<item 10>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 10>
=== END ITEM id=3e07a45e2a1026c0 ===

=== ITEM id=f7afc99926ad08ad ===
Gold sentences and facts:
Sentence 0: Catcha Lake is a Canadian lake located in the central part of Nova Scotia's Halifax Regional Municipality.
  Fact 0.0: Catcha Lake is a Canadian lake
  Fact 0.1: Catcha Lake is located in the central part of Nova Scotia's Halifax Regional Municipality
Sentence 1: The Hedley Mascot Mine was a gold mine in Hedley, British Columbia, Canada.
  Fact 1.0: The Hedley Mascot Mine was a gold mine
  Fact 1.1: The Hedley Mascot Mine was in Hedley, British Columbia, Canada

Retrieved items:
<item 1>
Leviathan Mine
Leviathan Mine is a United States superfund site (CERCLIS ID: CAD98067685) at an abandoned open-pit sulfur mine located in Alpine County, California.

 The mine is located at on the eastern slope of the Sierra Nevada at about 7,000 ft elevation, 6 mi east of Markleeville and 24 mi southeast of Lake Tahoe.

 The mine site comprises approximately 250 acre of land surrounded by the Toiyabe National Forest, which is only accessible a few months a year.

 The approximately 22 million tons of sulfur ore-containing crushed rock at the mine are responsible for contaminating the Leviathan and Aspen Creek, which join
with Mountaineer Creek to form Bryant Creek which ultimately empties into the East Fork of the Carson River.

 These water bodies are listed as 303(d) impaired.

 The site location is seismically active.
</item 1>

<item 2>
Graph query row:
entity: Halifax Regional Municipality
relationship: LOCATED_IN
related: Nova Scotia
</item 2>

<item 3>
Hedley, British Columbia
Hedley is an unincorporated town in southern British Columbia, Canada, named after Robert R. Hedley, the manager of the Hall Smelter in Nelson.

 Hedley is located at the foot of Nickel Plate Mountain in the Similkameen.

 The town had a population of approximately 400 as of 2005.

 In the early 1900s, Hedley's population peaked over 1,000 people, primarily due to the gold mining industry.

 The Hedley Museum and the Mascot Mine Museum display artifacts and photographs from this era.
</item 3>

<item 4>
Beaconsfield, Victoria
Beaconsfield is a satellite suburb of Melbourne, Victoria, Australia, 46 km south-east of Melbourne's Central Business District, located within the Shire of
Cardinia local government area.

 Beaconsfield recorded a population of 6,714 at the 2016 Census.
</item 4>

<item 5>
Hedley Mascot Mine
The Hedley Mascot Mine was a gold mine in Hedley, British Columbia, Canada.

 Gold was first discovered in the Nickel Plate Mountain area in 1897 and several small mines were developed over the years.

 The Hedley Mascot Mine operated between 1936 and 1949 and was one of the most unusual mining operations in the world, being built entirely on the side of a
mountain, 5,000 feet above the town of Hedley or seven thousand feet above sea level.

 In the 1990s, the British Columbia government was going to burn the site down because it posed a safety risk, but the Minister of Tourism at the time
intervened and, in 1995, steps were taken to preserve the site as a Provincial Heritage resource.
</item 5>

<item 6>
Catcha Lake
Catcha Lake is a Canadian lake located in the central part of Nova Scotia's Halifax Regional Municipality.
</item 6>

<item 7>
Graph query row:
entity: Catcha Lake
relationship: LOCATED_IN
related: Halifax Regional Municipality
</item 7>

<item 8>
Graph query row:
entity: Hedley Mascot Mine
relationship: LOCATED_IN
related: Hedley
</item 8>

<item 9>
Lake Shore Mine
The Lake Shore Mine is a gold mine located in Kirkland Lake, Ontario.

 In July 1912, Harry Oakes staked claims L-2605-6 which were in the lake itself and had reverted for non-performance of work.

 On September 6, 1912, he registered the transfer of claim L-1557 that Melville McDougall had staked for Oakes previously.

 On September 23, 1911, Harry Oakes registered the transfer of claim T-16635 from George Minaker and named the property Lake Shore.

 In production from 1918 to 1965, the mine produced almost 8.5 million ounces of gold and represented over a third of the gold produced in the entire camp.
</item 9>

<item 10>
Electoral district of Bass
The Electoral district of Bass is an electoral district of the Victorian Legislative Assembly.

 It covers an area of 1,383 sqkm south east of Melbourne, stretching from outer suburban Pakenham through rural areas to the coast at Inverloch and Phillip
Island.

 It includes the towns of Bass, Cape Patterson, Corinella, Cowes, Grantville, Inverloch, Kilcunda, Koo Wee Rup, Lang Lang, Rhyll, San Remo, Ventnor, Wonthaggi,
and part of suburban Pakenham.

 It lies within the Eastern Victoria Region of the upper house, the Legislative Council.
</item 10>
=== END ITEM id=f7afc99926ad08ad ===

=== ITEM id=7511228780e60566 ===
Gold sentences and facts:
Sentence 0: The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
  Fact 0.0: The International Grenfell Association (IGA) is an organization.
  Fact 0.1: The International Grenfell Association was founded by Sir Wilfred Grenfell.
  Fact 0.2: The International Grenfell Association provides health care.
  Fact 0.3: The International Grenfell Association provides education.
  Fact 0.4: The International Grenfell Association provides religious services.
  Fact 0.5: The International Grenfell Association provides rehabilitation.
  Fact 0.6: The International Grenfell Association provides other social activities.
  Fact 0.7: The International Grenfell Association serves fisherman and coastal communities in northern Newfoundland.
  Fact 0.8: The International Grenfell Association serves fisherman and coastal communities on the coast of Labrador.
Sentence 1: Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
  Fact 1.0: Sir Wilfred Thomason Grenfell held the title KCMG.
  Fact 1.1: Sir Wilfred Thomason Grenfell was born on 28 February 1865.
  Fact 1.2: Sir Wilfred Thomason Grenfell died on 9 October 1940.
  Fact 1.3: Sir Wilfred Thomason Grenfell was a medical missionary to Newfoundland.
  Fact 1.4: Sir Wilfred Thomason Grenfell was a medical missionary to Labrador.

Retrieved items:
<item 1>
Andrew Bryan (Baptist)
Andrew Bryan (1737–1812) founded First Bryan Baptist Church, affectionately called the Mother Church of Black Baptists, and First African Baptist Church of
Savannah in Savannah, Georgia, the first black Baptist churches to be established in America.

 Bryan was the former slave of Jonathan Bryan.
</item 1>

<item 2>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 2>

<item 3>
Edmund Robert Harris
Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris
Institute or Art School, Harris Technical School and the Harris Orphanage.
</item 3>

<item 4>
Training centre for release of the Atma-energy
Training centre for release of the Atma-energy (German: "Trainingszentrum zur Freisetzung der Atmaenergie"), also known as "Atman Foundation", was a new
religious movement active mainly on the island of Tenerife and in Germany.

 This sect was originally a splinter group of the Brahma Kumaris and is known for a police and media scare in which an alleged attempt to commit ritual suicide
took place in Teide National Park in Tenerife.

 The group believed in the end of the world but according to the religious studies scholar Georg Schmid and the sociologist Massimo Introvigne had no intention
of collective suicide.
</item 4>

<item 5>
International Grenfell Association
The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and
rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
</item 5>

<item 6>
Angelino Apelar
Angelino "Andy" Baldemor Apelar (June 16, 1927 - January 23, 2006) was an Evangelical Christian leader in the Filipino American community, and a founder of the
Association of Filipino Churches (AFC) of the Christian and Missionary Alliance.

 He had served as a missionary/pastor and radio broadcaster in the Philippines before moving to the United States to start the AFC.

 He served as president for several terms.

 He graduated from the Far Eastern Bible Institute and Seminary, where he also met his future wife, Purita Palomar, a member of one of Capiz province's
influential families.

 They have two children: Faith and Jemuel Apelar.
</item 6>

<item 7>
Ellis Gowing
Gowing was educated at Fort Street High School and the University of Sydney; and ordained in 1907.

 He began his career as a Curate at The Oaks, New South Wales, after which he served at Armidale Cathedral.

 Coming to England he was on the staff of St James the Less, Bethnal Green then domestic chaplain to the Bishop of Chelmsford from 1914 to 1917.

 He was vicar of Prittlewell from 1917 until his death; an honorary canon of Chelmsford Cathedral from 1921 to 1938 and Rural Dean of Canewdon from 1918 to
1938.

 He was given the Freedom of the County Borough of Southend-on-Sea in 1953.
</item 7>

<item 8>
John G. Wilson
John Guy Wilson (1842–1892) was a Manchester, UK patent agent and the founder of Wilson Gunn, one of the oldest firms of Patent and Trade Mark Attorneys in the
UK.

 John G. Wilson built up the business until his death in the 1890s.

 As a side line he acted as agent to the Alliance Insurance Company.

 There is no record that Mr. Wilson ever became a member of the Chartered Institute of Patent Agents.
</item 8>

<item 9>
Wilfred Grenfell
Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
</item 9>

<item 10>
Paul Schulte
Paul Schulte OMI (1896 – 1975), was a German priest and missionary, known as the "Flying Priest", who founded MIVA ("Missionary International Vehicular
Association") to provide automobiles, boats and airplanes for the service of missions throughout the world.
</item 10>
=== END ITEM id=7511228780e60566 ===

=== ITEM id=1ce3d691e60e9805 ===
Gold sentences and facts:
Sentence 0: Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.
  Fact 0.0: Paola Suárez was born on 23 June 1976.
  Fact 0.1: Paola Suárez is a retired professional tennis player.
  Fact 0.2: Paola Suárez is from Argentina.
Sentence 1:  She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.
  Fact 1.0: Paola Suárez was one of the most prominent women's doubles players throughout the early and mid-2000s.
  Fact 1.1: Paola Suárez won eight Grand Slam titles.
  Fact 1.2: All eight of Paola Suárez's Grand Slam titles were won with Virginia Ruano Pascual.
  Fact 1.3: Paola Suárez held the No. 1 doubles ranking for 87 non-consecutive weeks.
Sentence 2:  He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.
  Fact 2.0: Paola Suárez reached a career high singles ranking of 274 in July 2011.
  Fact 2.1: Paola Suárez is much better known as a leading doubles player with a huge serve.

Retrieved items:
<item 1>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 1>

<item 2>
1999 Italian Open – Women's Doubles
The 1999 Italian Open – Doubles was the doubles event of the fifty-fifth edition of the tennis tournament played at Rome, Italy, the most prestigious tennis
tournament in Southern Europe.

 It was the fifth WTA Tier I tournament of the year, and part of the European claycourt season.

 Virginia Ruano Pascual and Paola Suárez were the defending champions, but lost to Martina Hingis and Anna Kournikova in the quarterfinals.
</item 2>

<item 3>
1999 Copa Colsanitas – Doubles
The 1999 Copa Colsanitas – Doubles was the doubles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Janette Husárová and Paola Suárez were the defending champions but only Suárez competed that year with Laura Montalvo.
</item 3>

<item 4>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 4>

<item 5>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 5>

<item 6>
Paola Suárez
Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.

 She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano
Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.

 She was also a singles semifinalist at the 2004 French Open.
</item 6>

<item 7>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 7>

<item 8>
Samantha Stosur career statistics
This is a list of the main career statistics of professional Australian tennis player, Samantha Stosur.

 To date, Stosur has won eight WTA singles titles including one Grand Slam singles title at the 2011 US Open.

 She was also the runner-up at the 2010 French Open and a semi-finalist at the 2010 and 2011 WTA Tour Championships.

 Stosur has also enjoyed a successful doubles career, in which she has held the World No. 1 ranking and won twenty-four WTA doubles titles including two grand
slam women's doubles titles at the 2005 US Open and 2006 French Open and two year-ending championships at the 2005 and 2006 WTA Tour Championships.
</item 8>

<item 9>
1999 Brasil Open – Doubles
The 1999 Brasil Open – Doubles was the tennis doubles event of the first edition of the most prestigious tournament in Brazil.

 Argentinian team Laura Montalvo and Paola Suárez won the title, in what was the team's second title of the year, defeating Janette Husárová and Florencia Labat
in the final.
</item 9>

<item 10>
1999 Copa Colsanitas – Singles
The 1999 Copa Colsanitas – Singles was the singles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Paola Suárez was the defending champion but lost in the semifinals to Christína Papadáki.
</item 10>
=== END ITEM id=1ce3d691e60e9805 ===

=== ITEM id=f6e4d1a91bbe601e ===
Gold sentences and facts:
Sentence 0: Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.
  Fact 0.0: Remember the Titans is a 2000 American sports film
  Fact 0.1: Remember the Titans was produced by Jerry Bruckheimer
  Fact 0.2: Remember the Titans was directed by Boaz Yakin
Sentence 1: My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and Ray Walston, based on the 1960s television series of the same name in which Walston starred.
  Fact 1.0: My Favorite Martian is a 1999 American science fiction comedy film
  Fact 1.1: My Favorite Martian stars Christopher Lloyd
  Fact 1.2: My Favorite Martian stars Jeff Daniels
  Fact 1.3: My Favorite Martian stars Daryl Hannah
  Fact 1.4: My Favorite Martian stars Elizabeth Hurley
  Fact 1.5: My Favorite Martian stars Wallace Shawn
  Fact 1.6: My Favorite Martian stars Ray Walston
  Fact 1.7: My Favorite Martian is based on the 1960s television series of the same name
  Fact 1.8: Ray Walston starred in the 1960s television series My Favorite Martian
Sentence 2:  The film grossed $36.8 million domestically against a budget of $65 million.
  Fact 2.0: The film (My Favorite Martian) grossed $36.8 million domestically
  Fact 2.1: The film (My Favorite Martian) had a budget of $65 million

Retrieved items:
<item 1>
My Favorite Martian (film)
My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and
Ray Walston, based on the 1960s television series of the same name in which Walston starred.

 It was directed by Donald Petrie and written by Sherri Stoner and Deanna Oliver, based on the television series created by John L. Greene.

 Creatures were created by Amalgamated Dynamics from designs by Jordu Schell.

 The film grossed $36.8 million domestically against a budget of $65 million.
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
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 3>

<item 4>
List of accolades received by The Hurt Locker
Summit Entertainment then released "The Hurt Locker" to more screens.

 The film grossed $49.2 million worldwide, and was a success against its budget of $15 million.
</item 4>

<item 5>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 5>

<item 6>
Superman Returns
Superman Returns is a 2006 American superhero film directed and produced by Bryan Singer.

 It is based on the DC Comics character Superman and serves as an homage sequel to the motion pictures "Superman" (1978) and "Superman II" (1980), while
ignoring the events of "Superman III" (1983) and "" (1987).

 The film stars Brandon Routh as Clark Kent/Superman, Kate Bosworth as Lois Lane, Kevin Spacey as Lex Luthor, with James Marsden, Frank Langella, and Parker
Posey.

 The film tells the story of the title character returning to Earth after a five-year absence.

 He finds that his love interest Lois Lane has moved on with her life, and that his archenemy Lex Luthor is plotting a scheme that will destroy Superman and the
world.
</item 6>

<item 7>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 7>

<item 8>
Underdog (film)
Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.

 Directed by Frederik Du Chau and written by Joe Piscatella, Adam Rifkin, and Craig A. Williams, the film stars Jim Belushi, Peter Dinklage, John Slattery, and
Patrick Warburton with the voice talents of Jason Lee, Amy Adams, and Brad Garrett.

 Unlike the TV series, the Underdog character is portrayed as a regular dog rather than an anthropomorphic one.

 Underdog, voiced by Jason Lee, was played by a lemon beagle named Leo sporting a red sweater and a blue cape.

 The film grossed $65.3 million worldwide.
</item 8>

<item 9>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 9>

<item 10>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 10>
=== END ITEM id=f6e4d1a91bbe601e ===

=== ITEM id=e6b9e0cd9d191c1c ===
Gold sentences and facts:
Sentence 0: Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.
  Fact 0.0: Eric Arthur Blair was born on 25 June 1903.
  Fact 0.1: Eric Arthur Blair died on 21 January 1950.
  Fact 0.2: Eric Arthur Blair's pen name was George Orwell.
  Fact 0.3: Eric Arthur Blair is better known as George Orwell.
  Fact 0.4: Eric Arthur Blair was an English novelist.
  Fact 0.5: Eric Arthur Blair was an English essayist.
  Fact 0.6: Eric Arthur Blair was an English journalist.
  Fact 0.7: Eric Arthur Blair was an English critic.
Sentence 1: Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.
  Fact 1.0: Nelson Richard DeMille was born on August 23, 1943.
  Fact 1.1: Nelson Richard DeMille is an American author.
  Fact 1.2: Nelson Richard DeMille writes action adventure novels.
  Fact 1.3: Nelson Richard DeMille writes suspense novels.
Sentence 2:  DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
  Fact 2.0: Nelson Richard DeMille has written under the pen name Jack Cannon.
  Fact 2.1: Nelson Richard DeMille has written under the pen name Kurt Ladner.
  Fact 2.2: Nelson Richard DeMille has written under the pen name Ellen Kay.
  Fact 2.3: Nelson Richard DeMille has written under the pen name Brad Matthews.

Retrieved items:
<item 1>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 1>

<item 2>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 2>

<item 3>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 3>

<item 4>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 4>

<item 5>
The Panther (novel)
The Panther is a 2012 novel by American author Nelson DeMille.

 It is the sixth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "The Lion".

 The Panther is followed by DeMille’s 2015 novel, "Radiant Angel".

 Also featured in this novel is DeMille’s other fictional character, Paul Brenner, who appears in The General's Daughter and Up Country.
</item 5>

<item 6>
George Orwell
Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.

 His work is marked by lucid prose, awareness of social injustice, opposition to totalitarianism, and outspoken support of democratic socialism.
</item 6>

<item 7>
The Lion's Game
The Lion's Game is a 2000 novel by American author Nelson DeMille.

 It is the second of DeMille's novels to feature the detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New
York City.

 The 2000 novel The Lion’s Game is the sequel to Plum Island and is followed by the 2004 book Night Fall.

 The book also briefly mentions events from other DeMille novels like "The Charm School" and "The Gold Coast", that aren't strictly part of the Corey-universe.
</item 7>

<item 8>
The Lion (2010 novel)
The Lion is a 2010 novel by American author Nelson DeMille.

 It is the fifth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "Wild Fire".

 The Lion is followed by DeMille’s 2012 novel, "The Panther".
</item 8>

<item 9>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 9>

<item 10>
Wild Fire (novel)
Wild Fire is a 2006 novel by American author, Nelson DeMille.

 It is the fourth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to Night Fall and takes place approximately one year later.

 Wild Fire is followed by DeMille’s 2010 novel, The Lion.
</item 10>
=== END ITEM id=e6b9e0cd9d191c1c ===

=== ITEM id=7adea189d57baa70 ===
Gold sentences and facts:
Sentence 0: Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist of the twentieth century".
  Fact 0.0: Erving Goffman was born on 11 June 1922.
  Fact 0.1: Erving Goffman died on 19 November 1982.
  Fact 0.2: Erving Goffman was a Canadian-American sociologist.
  Fact 0.3: Erving Goffman was a writer.
  Fact 0.4: Some consider Erving Goffman the most influential American sociologist of the twentieth century.
Sentence 1:  In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens, Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
  Fact 1.0: In 2007 Erving Goffman was listed by The Times Higher Education Guide.
  Fact 1.1: The Times Higher Education Guide listed Goffman as the sixth most-cited author in the humanities and social sciences in 2007.
  Fact 1.2: In that 2007 listing, Anthony Giddens, Pierre Bourdieu and Michel Foucault were ranked ahead of Goffman.
  Fact 1.3: In that 2007 listing, Goffman was ranked ahead of Jürgen Habermas.
Sentence 2: Paul-Michel Foucault (15 October 1926 – 25 June 1984), generally known as Michel Foucault (] ), was a French philosopher, historian of ideas, social theorist, and literary critic.
  Fact 2.0: Paul-Michel Foucault was born on 15 October 1926.
  Fact 2.1: Paul-Michel Foucault died on 25 June 1984.
  Fact 2.2: Paul-Michel Foucault is generally known as Michel Foucault.
  Fact 2.3: Michel Foucault was a French philosopher.
  Fact 2.4: Michel Foucault was a historian of ideas.
  Fact 2.5: Michel Foucault was a social theorist.
  Fact 2.6: Michel Foucault was a literary critic.

Retrieved items:
<item 1>
Dramaturgy (sociology)
Dramaturgy is a sociological perspective commonly used in microsociological accounts of social interaction in everyday life.

 The term was first adapted into sociology from the theatre by Erving Goffman, who developed most of the related terminology and ideas in his 1959 book, "The
Presentation of Self in Everyday Life".

 Kenneth Burke, whom Goffman would later acknowledge as an influence, had earlier presented his notions of dramatism in 1945, which in turn derives from
Shakespeare.

 However, the fundamental difference between Burke's and Goffman's view is that Burke believed that life was in fact theatre, whereas Goffman viewed theatre as
a metaphor.

 If we imagine ourselves as directors observing what goes on in the theatre of everyday life, we are doing what Goffman called dramaturgical analysis, the study
of social interaction in terms of theatrical performance.
</item 1>

<item 2>
Identity Performance
In everyday interactions, the body serves as a critical site of identity performance.

 In conveying who we are to other people, we use our bodies to project information about ourselves.

 This is done through movement, clothes, speech, and facial expressions.

 What we put forward is our best effort at what we want to say about who we are.

 Yet while we intend to convey one impression, our performance is not always interpreted as we might expect.

 Through learning to make sense of others’ responses to our behavior, we can assess how well we have conveyed what we intended.

 We can then alter our performance accordingly.

 This process of performance, interpretation, and adjustment is what Erving Goffman calls impression management.

 Impression management is a part of a larger process where people seek to define a situation through their behavior.

 People seek to define social situations by using contextual cues from the environment around them.

 Social norms emerge from situational definitions, as people learn to read cues from the environment and the people present to understand what is appropriate
behavior.
</item 2>

<item 3>
Erving Goffman
Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist
of the twentieth century".

 In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens,
Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
</item 3>

<item 4>
Ray Birdwhistell
Collaborations with others, including initially Margaret Mead and Gregory Bateson, and later, Erving Goffman and Dell Hymes had huge influence on his work.

 For example, the book he is best known for, "Kinesics and Context", "would not have appeared if it had not been envisaged by Erving Goffman" and he explicitly
stated "the paramount and sustaining influence upon my work has been that of anthropological linguistics", a tradition most directly represented at the
University of Pennsylvania by Hymes.
</item 4>

<item 5>
Impression management
Impression management is a conscious or subconscious process in which people attempt to influence the perceptions of other people about a person, object or
event.

 They do so by regulating and controlling information in social interaction.

 It was first conceptualized by Erving Goffman in 1959, and then was expanded upon in 1967.

 An example of impression management theory in play is in sports such as soccer.

 At an important game, a player would want to showcase themselves in the best light possible, because there are college recruiters watching.

 This person would have the flashiest pair of cleats and try and perform their best to show off their skills.

 Their main goal may be to impress the college recruiters in a way that maximizes their chances of being chosen for a college team rather than winning the game.
</item 5>

<item 6>
Goffman (disambiguation)
Erving Goffman (1922–1982) was a Canadian sociologist
</item 6>

<item 7>
Total institution
A total institution is a place of work and residence where a great number of similarly situated people, cut off from the wider community for a considerable
time, together lead an enclosed, formally administered round of life.

 The concept is mostly associated with the work of sociologist Erving Goffman.
</item 7>

<item 8>
Identity management theory
Identity management theory (also frequently referred to as IMT) is an intercultural communication theory from the 1990s.

 It was developed by William R. Cupach and Tadasu Todd Imahori on the basis of Erving Goffman's "Interaction ritual: Essays on face-to-face behavior" (1967).

 Cupach and Imahori distinguish between intercultural communication (speakers from different cultures) and intracultural communication (speakers sharing the
same culture).
</item 8>

<item 9>
The Presentation of Self in Everyday Life
The Presentation of Self in Everyday Life is a 1956 sociology book by Erving Goffman, in which Goffman uses the imagery of the theatre in order to portray the
importance of human social interaction.

 Originally published in Scotland in 1956 and in the United States in 1959, it is Goffman’s first and most famous book, for which he received the American
Sociological Association’s MacIver award in 1961.
</item 9>

<item 10>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 10>
=== END ITEM id=7adea189d57baa70 ===
