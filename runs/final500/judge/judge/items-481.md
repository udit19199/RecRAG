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

=== ITEM id=523d107286d108e1 ===
Gold sentences and facts:
Sentence 0: Inner Secrets is the ninth studio album by Santana.
  Fact 0.0: Inner Secrets is a studio album by Santana.
  Fact 0.1: Inner Secrets is Santana's ninth studio album.
Sentence 1: Carlos Santana    (born July 20, 1947) is a Mexican and American musician who first became famous in the late 1960s and early 1970s with his band, Santana, which pioneered a fusion of rock and Latin American music.
  Fact 1.0: Carlos Santana was born on July 20, 1947.
  Fact 1.1: Carlos Santana is a Mexican musician.
  Fact 1.2: Carlos Santana is an American musician.
  Fact 1.3: Carlos Santana first became famous in the late 1960s and early 1970s.
  Fact 1.4: Carlos Santana first became famous with his band Santana.
  Fact 1.5: The band Santana pioneered a fusion of rock and Latin American music.

Retrieved items:
<item 1>
Inner Secrets
Inner Secrets is the ninth studio album by Santana.

 It was released in 1978 and marks the start of the phase of Santana's career where he moved away from the fusion of Latin, jazz, rock and blues that marked his
previous records and began to move towards an album-oriented rock direction.

 As such, the album's quality is widely disputed among fans. "

Stormy" and "One Chain (Don't Make No Prison)" were both hit singles.

 In The Netherlands "Well All Right" was released as a single and reached #22 in the top 40.
</item 1>

<item 2>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 2>

<item 3>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 3>

<item 4>
Carlos Santana
Carlos Santana    (born July 20, 1947) is a Mexican and American musician who first became famous in the late 1960s and early 1970s with his band, Santana,
which pioneered a fusion of rock and Latin American music.

 The band's sound featured his melodic, blues-based guitar lines set against Latin and African rhythms featuring percussion instruments such as timbales and
congas not generally heard in rock music.

 Santana continued to work in these forms over the following decades.

 He experienced a resurgence of popularity and critical acclaim in the late 1990s.

 In 2003 "Rolling Stone" magazine listed Santana at number 20 on their list of the 100 Greatest Guitarists of All Time.

 He has won 10 Grammy Awards and three Latin Grammy Awards.
</item 4>

<item 5>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 5>

<item 6>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 6>

<item 7>
Information (Berlin album)
Information is the first album by Berlin, released in 1980 by Vinyl Records.

 It was recorded during the period when Terri Nunn had temporarily left the band to pursue an acting career and Virginia Macolino performed the lead vocals.

 Several songs were written by previous lead singer Toni Childs who went solo prior to the album's release.
</item 7>

<item 8>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 8>

<item 9>
Mike Oldfield
Michael Gordon Oldfield (born 15 May 1953) is an English musician and composer.

 His work blends progressive rock with world, folk, classical, electronic, ambient, and new-age music.

 His biggest commercial success is the 1973 album "Tubular Bells"which launched Virgin Records and became a hit in America after its opening was used as the
theme for the film "The Exorcist".

 He recorded the 1983 hit single "Moonlight Shadow" and a rendition of the Christmas piece "In Dulci Jubilo".
</item 9>

<item 10>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 10>
=== END ITEM id=523d107286d108e1 ===

=== ITEM id=b7fa0c18700b735e ===
Gold sentences and facts:
Sentence 0: The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna (Tenerife, Canary Islands, Spain).
  Fact 0.0: The Church of the Guanche People's Spanish name is 'Iglesia del Pueblo Guanche'.
  Fact 0.1: The Church of the Guanche People is a religious organisation.
  Fact 0.2: The Church of the Guanche People was founded in 2001.
  Fact 0.3: The Church of the Guanche People was founded in the city of San Cristobal de La Laguna.
  Fact 0.4: San Cristobal de La Laguna is in Tenerife, Canary Islands, Spain.
Sentence 1:  Tenerife is the largest and most populous island of Macaronesia.
  Fact 1.0: Tenerife is the largest island of Macaronesia.
  Fact 1.1: Tenerife is the most populous island of Macaronesia.

Retrieved items:
<item 1>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 1>

<item 2>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 2>

<item 3>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 3>

<item 4>
Gran i General Consell
The Gran i General Consell (Catalan: "Great and General Council") was the supreme political, administrative, and representative organ of the Kingdom of Majorca.

 Since the Kingdom of Majorca did not have courts, the Gran i General Consell took over most of the functions they would otherwise have had, including the role
of a representative body.

 The Gran i General Consell evolved from the Catalan municipal councils, especially that of the City of Majorca (present-day Palma de Mallorca.

 The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova
Planta Decree of Majorca and Ibiza (28 November 1715).

 At the same time, all other separate Majorcan institutions were dissolved.
</item 4>

<item 5>
Chiguata District
Chiguata is a district located in the Arequipa province, 30 km away from the city of Arequipa, Peru, on the slopes of the volcano Pikchu Pikchu (where at its
summit pre-Inca worshipped remains and artifacts have been found).

 The name "chiguata" comes from the Quechua words "chiri" meaning "cold" and "wata" meaning "year" (cold all year).

 Ancient Chiguata was inhabited by the local natives until the arrival of Spanish influence in charge of the emissary Diego Hernández de Córdova in 1540.

 Soon after, the construction of a small plaza was initiated, and according to the Dominican religious order and local sources, the plaza's "Templo del Espíritu
Santo" ("Sacred Spirit temple") along with its domes and lateral cover was not begun until 1739.

 The design of the temple is quite interesting, with two angels in feathered nightwear and their arms lifted vertically, separated by twelve fringes of
rectangular flowers which add to the whole decoration of the temple.
</item 5>

<item 6>
Guanche mummies
Guanche mummies are the intentionally desiccated remains of members of the indigenous Guanche people of the Canary Islands.

 The majority of Guanche mummies were made during the eras prior to Spanish settlement of the area in the 15th century.

 The methods of embalming are similar to those that were used by the Ancient Egyptians, though fewer mummies remain from the Guanche due to looting and
desecration.
</item 6>

<item 7>
Chiguata District
Cherubs also decorate the circular cornice, along with four saints in full-length reliefs that occupy the "pechinas" of the columns that sustain the temple's
dome, which is considered one of the most original works of the colony and now visited by occasional tourists.
</item 7>

<item 8>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 8>

<item 9>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 9>

<item 10>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 10>
=== END ITEM id=b7fa0c18700b735e ===

=== ITEM id=a3117efd3e9dda87 ===
Gold sentences and facts:
Sentence 0: The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of the Stones River Campaign in the Western Theater of the American Civil War.
  Fact 0.0: The Battle of Stones River is also called the Second Battle of Murfreesboro.
  Fact 0.1: The Battle of Stones River was fought from December 31, 1862, to January 2, 1863.
  Fact 0.2: The Battle of Stones River was fought in Middle Tennessee.
  Fact 0.3: The Battle of Stones River was the culmination of the Stones River Campaign.
  Fact 0.4: The Stones River Campaign took place in the Western Theater of the American Civil War.
Sentence 1: The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.
  Fact 1.0: The Battle of the Ch'ongch'on River is also known as the Battle of the Ch'ongch'on.
  Fact 1.1: The Battle of the Ch'ongch'on River is also known as the Second Phase Campaign Western Sector.
  Fact 1.2: The Battle of the Ch'ongch'on River was a decisive battle in the Korean War.
  Fact 1.3: The Battle of the Ch'ongch'on River took place from November 25 to December 2, 1950.
  Fact 1.4: The Battle of the Ch'ongch'on River took place along the Ch'ongch'on River Valley.
  Fact 1.5: The Ch'ongch'on River Valley is in the northwestern part of North Korea.

Retrieved items:
<item 1>
Battle of Piqua
The Battle of Piqua, also known as the Battle of Pekowee or Pekowi, was part of the western campaign during the American Revolutionary War.

 Led by General George Rogers Clark, over 1,000 soldiers (among them Daniel Boone and Simon Kenton) crossed the Ohio River near present-day Cincinnati and
burned five Shawnee villages, including Old Chillicothe, along the Little Miami River.

 Peter Loramie's Store, a British trading post-located in what was later Fort Loramie, Ohio in Shelby County, Ohio-, was also burned by Clark's men.

 The Shawnee gradually withdrew during the first few days before finally engaging American forces 7 miles west of Springfield, Ohio on August 8, 1780.

 Joseph Rogers, a cousin of George Rogers Clark, had previously accompanied him to Kentucky and was later captured by the Shawnee near Maysville.

 Despite having been adopted by the tribe, he was killed during the battle while trying to join American forces.
</item 1>

<item 2>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 2>

<item 3>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 3>

<item 4>
Battle of Stones River
The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of
the Stones River Campaign in the Western Theater of the American Civil War.

 Of the major battles of the Civil War, Stones River had the highest percentage of casualties on both sides.

 Although the battle itself was inconclusive, the Union Army's repulse of two Confederate attacks and the subsequent Confederate withdrawal were a much-needed
boost to Union morale after the defeat at the Battle of Fredericksburg, and it dashed Confederate aspirations for control of Middle Tennessee.
</item 4>

<item 5>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 5>

<item 6>
The entries span Revolutionary War figures, battles, and units; a Scottish clan battle; a 17th-century Ukrainian-Polish battle; and a Civil War regiment.
Benjamin Logan led Kentucky militia and statehood efforts; his brother John served in Virginia politics and became Kentucky’s first treasurer. Frontier
conflicts included the 1780 Westervelt massacre, 1782 Blue Licks defeat, and Clark’s 1780 Piqua campaign. Other topics include Yorktown, Germantown, the
</item 6>

<item 7>
Battle of Port Republic
The Battle of Port Republic was fought on June 9, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Port Republic was a fierce contest between two equally determined foes and was the most costly battle fought by Jackson's Army of the Valley during its
campaign.

 Together, the battles of Cross Keys (fought the previous day) and Port Republic were the decisive victories in Jackson's Valley Campaign, forcing the Union
armies to retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 7>

<item 8>
Second Battle of Fort Wagner
The Second Battle of Fort Wagner, also known as the Second Assault on Morris Island or the Battle of Fort Wagner, Morris Island, was fought on July 18, 1863,
during the American Civil War.

 Union Army troops commanded by Brig. Gen. Quincy Gillmore, launched an unsuccessful assault on the Confederate fortress of Fort Wagner, which protected Morris
Island, south of Charleston Harbor.

 The battle came one week after the First Battle of Fort Wagner.
</item 8>

<item 9>
Battle of Cross Keys
The Battle of Cross Keys was fought on June 8, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Together, the battles of Cross Keys and Port Republic the following day were the decisive victories in Jackson's Valley Campaign, forcing the Union armies to
retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 9>

<item 10>
The text covers Civil War battles, campaigns, and orders of battle. In June 1862, Stonewall Jackson’s victories at Cross Keys and Port Republic enabled him to
join Robert E. Lee near Richmond. Lee’s Seven Days Battles (June 25–July 1) drove McClellan’s Union army from Richmond; Gaines’s Mill and Malvern Hill are
highlighted. Later events include Lee’s Northern Virginia and Maryland campaigns, Antietam (22,717 casualties), Stones River
</item 10>
=== END ITEM id=a3117efd3e9dda87 ===

=== ITEM id=baa2c1bd51e971da ===
Gold sentences and facts:
Sentence 0: The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.
  Fact 0.0: The Livesey Hall War Memorial commemorates the fallen of World War I.
  Fact 0.1: The Livesey Hall War Memorial commemorates the fallen of World War II.
  Fact 0.2: The commemorated fallen had been employed by the South Suburban Gas Company of London.
Sentence 1:  Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).
  Fact 1.0: Over 60 million people were killed.
  Fact 1.1: The deaths were about 3% of the 1940 world population.
  Fact 1.2: The estimated 1940 world population was 2.3 billion.

Retrieved items:
<item 1>
Livesey Hall War Memorial
The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.

 It is also a tribute to those employees who served in the wars.

 The monument was designed and executed by British sculptor Sydney March, of the March family of artists.
</item 1>

<item 2>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 2>

<item 3>
Vietnam War Memorial, Hanoi
The War Memorial in Hanoi is located across the Ba Dinh Square, across the Ho Chi Minh Mausoleum and close to Hanoi Citadel.

 Constructed in 1993 in a fusion of traditional Vietnamese and modernist architecture, the memorial commemorates men and women who sacrificed themselves during
the Second Indochina War.

 The war is known by many names, e.g. as the American War in Vietnam.

 The memorial is a focal point for state functions, commemorating the war dead.
</item 3>

<item 4>
Rainham War Memorial
The Rainham War Memorial commemorates soldiers killed in both World Wars as well as civilian casualties of World War II.
</item 4>

<item 5>
Alamein Memorial
The Alamein Memorial is a Commonwealth War Graves Commission war memorial in the El Alamein War Cemetery, El Alamein, Egypt.

 The memorial commemorates 11,866 Commonwealth forces members who died during World War II.

 The memorial was designed by Hubert Worthington and unveiled by Viscount Montgomery of Alamein on 24 October 1954.
</item 5>

<item 6>
World War II casualties
World War II was the deadliest military conflict in history in absolute terms of total casualties.

 Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).

 The tables below give a detailed country-by-country count of human losses.

 World War II fatality statistics vary, with estimates of total deaths ranging from 50 million to more than 80 million.

 The higher figure of over 80 million includes deaths from war-related disease and famine.

 Civilians killed totalled 50 to 55 million, including 19 to 28 million from war-related disease and famine.

 Military deaths from all causes totalled 21 to 25 million, including deaths in captivity of about 5 million prisoners of war.
</item 6>

<item 7>
Tower Hill Memorial
The Tower Hill Memorial is a Commonwealth War Graves Commission war memorial on the south side of Trinity Square Gardens, in London, England.

 The memorial commemorates those from the Merchant Navy and fishing fleets who died during both world wars and have "no grave but the sea".

 The memorial was designed by Edwin Lutyens with sculpture work by William Reid Dick, the Second World War extension was designed by Edward Maufe with sculpture
work by Charles Wheeler.
</item 7>

<item 8>
Portsmouth Naval Memorial
The Portsmouth Naval Memorial, sometimes known as Southsea Naval Memorial, is a war memorial in Portsmouth, Hampshire, England, on Southsea Common beside
Clarence Esplanade, between Clarence Pier and Southsea Castle.

 The memorial commemorates approximately 25,000 British and Commonwealth sailors who were lost in the World Wars, around 10,000 sailors in the First World War
and 15,000 in the Second World War.

 The memorial features a central obelisk, with names of the dead on bronze plaques arranged around the memorial according to the year of death.
</item 8>

<item 9>
Freetown Memorial
The Freetown Memorial is a Commonwealth War Graves Commission war memorial located outside the Secretariate Building in Freetown, Sierra Leone.

 The memorial generally commemorates Soldiers from Sierra Leone who died whilst serving with the Royal West African Frontier Force in West Africa and whose
graves are not known.

 The Freetown Memorial commemorates 1,109 First World War casualties and 247 from the Second World War.
</item 9>

<item 10>
Arras Flying Services Memorial
The Arras Flying Services Memorial Commonwealth War Graves Commission war memorial in the Faubourg d'Amiens Cemetery, Arras, France.

 The memorial commemorates nearly 1,000 airmen from forces of the Commonwealth who were killed on the Western Front during World War I and who have no known
grave.

 The memorial was designed by Edwin Lutyens, sculpted by William Reid Dick and unveiled by Hugh Trenchard, 1st Viscount Trenchard, Marshal of the Royal Air
Force on 31 July 1932.
</item 10>
=== END ITEM id=baa2c1bd51e971da ===

=== ITEM id=f6d599d6ee2e8af2 ===
Gold sentences and facts:
Sentence 0: Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
  Fact 0.0: Carly Gullickson was born on November 26, 1986.
  Fact 0.1: Carly Gullickson is a former American professional tennis player.
Sentence 1: Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.
  Fact 1.0: Francesca Schiavone was born on 23 June 1980 in Milan.
  Fact 1.1: Francesca Schiavone is an Italian tennis player.
  Fact 1.2: Francesca Schiavone turned professional in 1998.

Retrieved items:
<item 1>
Nicole Strausak
Nicole Strausak (born October 21, 1971) is a Swiss curler who retired from competitive curling in March 2005.

 Amongst other awards, she received a World Junior Silver medal in 1991, a European Silver medal in 1993 and a World Silver medal in 2000.

 In 1999, at the European Curling Championships in Chamonix, as well as receiving a Bronze medal, she was awarded the Fair Play & Friendly Award by her fellow
athletes for her conduct and attitude, on and off the ice.
</item 1>

<item 2>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 2>

<item 3>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 3>

<item 4>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 4>

<item 5>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 5>

<item 6>
2010 US Open – Mixed Doubles
Carly Gullickson and Travis Parrott were the defending champions, but lost in the first round to Gisela Dulko and Pablo Cuevas 2–6, 4–6.
</item 6>

<item 7>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 7>

<item 8>
2006 Qatar Ladies Open – Doubles
Alicia Molik and Francesca Schiavone were the defending champions, but Molik did not compete this year.

 Schiavone teamed up with Květa Peschke and lost in querterfinals to Elena Likhovtseva and Vera Zvonareva.
</item 8>

<item 9>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 9>

<item 10>
2006 Generali Ladies Linz – Doubles
Gisela Dulko and Květa Peschke were the defending champions, but competed this year with different partners.

 Dulko teamed up with Michaëlla Krajicek and lost in first round to Eleni Daniilidou and Jasmin Wöhr, while Peschke teamed up with Francesca Schiavone and had
to withdrew in their semifinal match (against Corina Morariu and Katarina Srebotnik) as Schiavone had her right knee injured.
</item 10>
=== END ITEM id=f6d599d6ee2e8af2 ===

=== ITEM id=e638d94888b3b986 ===
Gold sentences and facts:
Sentence 0: Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.
  Fact 0.0: Samantha Cristoforetti was born 26 April 1977 in Milan.
  Fact 0.1: Samantha Cristoforetti is an Italian European Space Agency astronaut.
  Fact 0.2: Samantha Cristoforetti is an Italian Air Force pilot.
  Fact 0.3: Samantha Cristoforetti is an Italian engineer.
Sentence 1: Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force), and a former CNES spationaut.
  Fact 1.0: Jean-Loup Jacques Marie Chrétien was born 20 August 1938.
  Fact 1.1: Jean-Loup Jacques Marie Chrétien is French.
  Fact 1.2: Jean-Loup Jacques Marie Chrétien is retired.
  Fact 1.3: Jean-Loup Jacques Marie Chrétien holds the rank of Général de Brigade (brigadier general).
  Fact 1.4: Jean-Loup Jacques Marie Chrétien served as a general in the Armée de l'Air (French air force).
  Fact 1.5: Jean-Loup Jacques Marie Chrétien is a former CNES spationaut.

Retrieved items:
<item 1>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 1>

<item 2>
The profiles cover NASA computer scientist Annie Easley and space espresso: ISSpresso’s first in-space coffee, brewed by Samantha Cristoforetti in 2015. They
describe astrophysicist Jean-Loup Puget; missions Soyuz TM-7/T-6 and French astronaut Jean-Loup Chrétien; Patrick Baudry; Project Highwater; Apollo 7;
SpaceShipOne’s first privately funded human spaceflight; the Soyuz rocket; Lockheed test pilot John Christiansen
</item 2>

<item 3>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 3>

<item 4>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 4>

<item 5>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 5>

<item 6>
Jean-Loup Puget
Jean-Loup Puget (born 7 March 1947) is a French astrophysicist.

 His current research interests lie in the Cosmic Microwave Background.

 Jean-Loup Puget and his collaborators reported the first identification of the Cosmic infrared background using COBE data.

 He is also, along with Alain Léger, credited with the origin of the hypothesis that the series of infrared lines observed in numerous astrophysical objects are
caused by emission from polycyclic aromatic hydrocarbons.

 He is currently principal investigator of the HFI module of the Planck space mission.
</item 6>

<item 7>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 7>

<item 8>
Samantha Cristoforetti
Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.

 She holds the record for the longest uninterrupted spaceflight of a European astronaut (199 days, 16 hours), and until June 2017 held the record for the
longest single space flight by a woman until this was broken by Peggy Whitson.

 She is also the first Italian woman in space.

 Samantha Cristoforetti is also known as the first person who brewed an espresso coffee in space.
</item 8>

<item 9>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 9>

<item 10>
Soyuz T-6
Soyuz T-6 was a manned spaceflight to Earth orbit to the Salyut 7 space station in 1982.

 Along with two Soviet cosmonauts, the crew included a Frenchman, Jean-Loup Chrétien.
</item 10>
=== END ITEM id=e638d94888b3b986 ===

=== ITEM id=23e9ca3a3c5ff058 ===
Gold sentences and facts:
Sentence 0:  is a board game created by Glenn Drover and published by Eagle Games in 2003.
  Fact 0.0: The board game in this sentence was created by Glenn Drover.
  Fact 0.1: The board game in this sentence was published by Eagle Games in 2003.
Sentence 1: Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.
  Fact 1.0: Can't Stop is a board game designed by Sid Sackson.
  Fact 1.1: Can't Stop was originally published by Parker Brothers in 1980.
  Fact 1.2: Can't Stop was long out of print in the United States.

Retrieved items:
<item 1>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 1>

<item 2>
Atari Corporation, formed in 1984 after Jack Tramiel bought Atari’s computer and console divisions, made systems including the ST, 7800, Lynx, and Jaguar before
merging with JTS in 1996; Hasbro acquired its IP in 1998. Atari-related computing included Megamax C, a fast ST compiler; TOS/GEM and efforts toward
multitasking successors; Happy disk-drive upgrades; ATASCII for 8-bit computers; and the ST, an early
</item 2>

<item 3>
TriBond (1990) is a knowledge-and-reasoning board game sold in 14 countries. Tapp-Tarock, an Austrian three-player tarot game recorded in 1821, was once popular
in Viennese cafés. Three-card brag, a 16th-century British bluffing game descended from Primero, helped inspire poker. Triomphe, a late-15th-century
trick-taking game, introduced randomly chosen trumps and inspired many later games. Ombre
</item 3>

<item 4>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 4>

<item 5>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 5>

<item 6>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 6>

<item 7>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 7>

<item 8>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 8>

<item 9>
Francis Tresham (game designer)
Francis Tresham is a United Kingdom-based board game designer who has been producing board games since the early 1970s.

 Tresham founded and ran games company Hartland Trefoil (founded 1971), a company well known for its "Civilization" board game, until its sale to MicroProse in
1997.

 His "1829" game was the first of the "18xx" board game series and some of his board games have inspired Sid Meier computer games such as "Railroad Tycoon".
</item 9>

<item 10>
The entries cover games across several eras and genres: Japan-only 1989 Famicom game *A Week of Garfield*, based on the comic; *Kill Doctor Lucky*, a humorous
1996 board game that won a 1997 Origins Award; Sid Sackson’s dice-risk racing game *Can’t Stop* (1980; reprinted 2007, iOS 2012); 2003 light war game *Attack!*,
also an Origins Award winner; the luck-based
</item 10>
=== END ITEM id=23e9ca3a3c5ff058 ===

=== ITEM id=73a2abf39efc0c5c ===
Gold sentences and facts:
Sentence 0: Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
  Fact 0.0: Scott Treleaven is a Canadian artist.
  Fact 0.1: Scott Treleaven's work employs a variety of media.
  Fact 0.2: The media employed in Scott Treleaven's work include collage, film, video, drawing, photography and installation.
Sentence 1: Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.
  Fact 1.0: Edmund Mortimer's dates are 21 August 1874 to 21 May 1944.
  Fact 1.1: Edmund Mortimer was an American actor.
  Fact 1.2: Edmund Mortimer was an American film director.

Retrieved items:
<item 1>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 1>

<item 2>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 2>

<item 3>
Tommy Atkins (director)
Tommy Atkins was an American director of the silent and early sound film eras.

 Born on July 18, 1887 in Springfield, Massachusetts, he would make his entrance into the film industry as the assistant director to Ralph Ince on the 1920
silent film, "Out of the Snows".

 It would be another eight years before he would make another film, again as assistant director, this time for FBO Pictures, on another silent film, "Crooks
Can't Win".

 He'd work as the assistant director on another sixteen films between 1928 and 1934, the most notable of which would be 1933's "Morning Glory", directed by
Lowell Sherman and starring Katharine Hepburn and Douglas Fairbanks Jr..

 In 1934 he would be given the chance to helm his first picture, "The Silver Streak", which was one of the top money-makers for RKO Pictures that year.

 He would only direct two more films, the second of which, "Hi, Gaucho!

", he would also write the story for.
</item 3>

<item 4>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 4>

<item 5>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 5>

<item 6>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 6>

<item 7>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 7>

<item 8>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 8>

<item 9>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 9>

<item 10>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 10>
=== END ITEM id=73a2abf39efc0c5c ===

=== ITEM id=e438db1b9ff31dc2 ===
Gold sentences and facts:
Sentence 0: Haunted is the second album by American singer-songwriter Poe, released in 2000 (see 2000 in music) after a five-year hiatus from her debut album "Hello" in 1995.
  Fact 0.0: Haunted is the second album by American singer-songwriter Poe.
  Fact 0.1: Haunted was released in 2000.
  Fact 0.2: Haunted was released after a five-year hiatus from Poe's debut album 'Hello', which came out in 1995.
Sentence 1:  The self-produced album was created as a tribute to her father, and counterpart to her brother Mark Z. Danielewski's novel "House of Leaves".
  Fact 1.0: Haunted was self-produced.
  Fact 1.1: Haunted was created as a tribute to Poe's father.
  Fact 1.2: Haunted was created as a counterpart to the novel 'House of Leaves' by Poe's brother Mark Z. Danielewski.
Sentence 2:  A bestseller, it has been translated into a number of languages, and is followed by a companion piece, "The Whalestoe Letters".
  Fact 2.0: Haunted is a bestseller.
  Fact 2.1: Haunted has been translated into a number of languages.
  Fact 2.2: Haunted is followed by a companion piece, 'The Whalestoe Letters'.

Retrieved items:
<item 1>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 1>

<item 2>
The Spectre Knight
The Spectre Knight is a one-act "fanciful operetta" with a libretto by James Albery and music by Alfred Cellier.

 It was first performed on 9 February 1878 at the Opera Comique by the Comedy Opera Company as a companion piece to "The Sorcerer".

 The piece continued to run until 23 March 1878 and was revived by the company from 28 May 1878 to 10 August 1878 as a companion piece to "H.M.S. Pinafore".

 The piece had a run in New York in 1880 and was toured in Britain and America.
</item 2>

<item 3>
The Pig, or Václav Havel's Hunt for a Pig
The Pig, or Václav Havel's Hunt for a Pig is the final work by Václav Havel, co-authored by Vladimír Morávek.

 The English translation is by Edward Einhorn .

 Originally a short dialogue from 1987 (entitled simply "The Pig") and printed in a samizdat, the piece is a comic (and true) story of Václav Havel’s efforts to
hold a pig roast for his friends.
</item 3>

<item 4>
Haunted (Poe album)
Haunted is the second album by American singer-songwriter Poe, released in 2000 (see 2000 in music) after a five-year hiatus from her debut album "Hello" in
1995.

 The self-produced album was created as a tribute to her father, and counterpart to her brother Mark Z. Danielewski's novel "House of Leaves".
</item 4>

<item 5>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 5>

<item 6>
Henry James
Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary
modernism, and is considered by many to be among the greatest novelists in the English language.

 He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
</item 6>

<item 7>
A Predicament
"A Predicament" is a humorous short story by Edgar Allan Poe, usually combined with its companion piece "How to Write a Blackwood Article."

 It was originally titled "The Scythe of Time".

 The paired stories parody the Gothic sensation tale, popular in England and America since the early 19th century.
</item 7>

<item 8>
The Frost King
"The Frost King" was a short story about King Jack Frost written by 11-year-old Helen Keller.

 Keller's teacher, Anne Sullivan, had mentioned that the autumn leaves were "painted ruby, emerald, gold, crimson, and brown," and Keller, by her own account,
imagined fairies doing the work.

 Keller wrote a story about how a cask of jewels, being transported by fairy servants, had melted in the sun and covered the leaves.

 As a birthday gift, Keller sent the story to Michael Anagnos, head of the Perkins School for the Blind, who published the story in "The Mentor," the Perkins
alumni magazine.

 It was picked up by "The Goodson Gazette", a journal on deaf-blind education, based in Virginia.
</item 8>

<item 9>
James Poe
James Poe (October 4, 1921 – January 24, 1980) was an American film and television screenwriter.

 He is best known for his work on the movies "Around the World in 80 Days" for which he jointly won an Academy Award in 1956, "Cat on a Hot Tin Roof", "Summer
and Smoke", "Lilies of the Field", and "They Shoot Horses, Don't They?

".
</item 9>

<item 10>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 10>
=== END ITEM id=e438db1b9ff31dc2 ===

=== ITEM id=8915ccbdb129a4ea ===
Gold sentences and facts:
Sentence 0:  The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI Studios in London, which was soon renamed Abbey Road Studios.
  Fact 0.0: The title and cover are an homage to the Beatles album Abbey Road.
  Fact 0.1: 926 East McLemore Avenue is the address of the Stax Studios in Memphis.
  Fact 0.2: Abbey Road was the address of EMI Studios in London.
  Fact 0.3: EMI Studios in London was soon renamed Abbey Road Studios.
Sentence 1:  This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
  Fact 1.0: This road is best known for the Abbey Road Studios.
  Fact 1.1: This road is best known for the 1969 album Abbey Road by The Beatles.

Retrieved items:
<item 1>
Westminster tube station
Westminster is a London Underground station in the City of Westminster.

 It is served by the Circle, District and Jubilee lines.

 On the Circle and District lines, the station is between St. James's Park and Embankment, and on the Jubilee line it is between Green Park and Waterloo.

 It is in Travelcard Zone 1.

 The station is located at the corner of Bridge Street and Victoria Embankment and is close to the Houses of Parliament, Westminster Abbey, Parliament Square,
Whitehall, Westminster Bridge, and the London Eye.

 Also close by are Downing Street, the Cenotaph, Westminster Millennium Pier, the Treasury, the Foreign and Commonwealth Office, and the Supreme Court.
</item 1>

<item 2>
Abbey Road, London
Abbey Road is a thoroughfare in the borough of Camden and the City of Westminster in London, running roughly northwest to southeast through St. John's Wood,
near Lord's Cricket Ground.

 It is part of the B507 road.

 This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
</item 2>

<item 3>
Doughty Street
Doughty Street is a broad tree lined street in the Holborn district of the London Borough of Camden.

 The southern part is a continuation of the short John Street, which comes off Theobald's Road.

 The northern part crosses Guilford Street and ends at Mecklenburgh Square.

 The street is named after a landlord of the area at the time it was built, Henry Doughty.
</item 3>

<item 4>
Come Together
"Come Together" is a song by the Beatles written primarily by John Lennon and credited to Lennon–McCartney.

 The song is the opening track on the album "Abbey Road" and was also released as a single coupled with "Something".

 The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
</item 4>

<item 5>
Westminster St Margaret and St John
St Margaret was an ancient parish in the City and Liberty of Westminster and the county of Middlesex.

 It included the core of modern Westminster, including the Palace of Westminster and the area around, but not including Westminster Abbey.

 In 1727 it was divided into St Margaret's and St John's, to coincide with the building of the Church of St John the Evangelist, constructed by the Commission
for Building Fifty New Churches in Smith Square to meet the demands of the growing population, but there continued to be a single vestry for the parishes of St
Margaret and St John.

 This was reformed in 1855 by the Metropolis Management Act, and the two parishes formed the Westminster District until 1887.

 In 1889 St Margaret and St John became part of the County of London.

 The vestry was abolished in 1900, to be replaced by Westminster City Council, but St Margaret and St John continued to have a nominal existence until 1922.
</item 5>

<item 6>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 6>

<item 7>
City of Westminster Magistrates' Court
The City of Westminster Magistrates' Court was a magistrates' court located at 70 Horseferry Road, in the City of Westminster, London.

 It was originally called Horseferry Road Magistrates' Court, after the road in which it was sited.

 However, it was renamed in July 2006 following the closure of Bow Street Magistrates' Court.

 It served as the court where the Chief Magistrate of England and Wales sat, and all extradition and terrorism-related cases passed through the court.

 The court closed permanently on 22 September 2011, and was replaced on 27 September 2011 with Westminster Magistrates' Court, built on the site of Marylebone
Magistrates' Court at 181 Marylebone Road.
</item 7>

<item 8>
Temple Bar, London
Temple Bar was the principal ceremonial entrance to the City of London on its western side from the City of Westminster.

 It is situated on the historic royal ceremonial route from the Tower of London to the Palace of Westminster, the two chief residences of the mediaeval English
monarchs, and from the Palace of Westminster to St Paul's Cathedral.

 The road east of Temple Bar and within the City is Fleet Street, the road to the west, in Westminster, is The Strand.

 At Temple Bar the Corporation of the City of London formerly erected a barrier to regulate trade into the City.

 The 19th century Royal Courts of Justice are located next to it on its north side, having been moved from Westminster Hall.

 To its south is the Temple Church and the Inner Temple and Middle Temple Inns of Court.

 As the most important entrance to the City of London from Westminster, it was formerly long the custom for the monarch to halt at Temple Bar before entering
the City of London, in order for the Lord Mayor to offer the Corporation's pearl-encrusted Sword of State as a token of loyalty.
</item 8>

<item 9>
Westminster
Westminster ( ) is an area of central London within the City of Westminster, part of the West End, on the north bank of the River Thames.

 Westminster's concentration of visitor attractions and historic landmarks, one of the highest in London, includes the Palace of Westminster, Buckingham Palace,
Westminster Abbey and Westminster Cathedral.
</item 9>

<item 10>
Temple Bar, London
The term "Temple Bar" strictly refers to a notional bar or barrier across the route, but is commonly used to refer to the 17th-century ornamental Baroque arched
gateway designed by Christopher Wren which spanned the road until its removal in 1878.

 Wren's arch was preserved and was re-erected in 2004 in the City, in Paternoster Square next to St Paul's Cathedral.
</item 10>
=== END ITEM id=8915ccbdb129a4ea ===

=== ITEM id=2937f34f656c6584 ===
Gold sentences and facts:
Sentence 0: Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
  Fact 0.0: Gunnar Eric Nelson was born on September 20, 1967.
  Fact 0.1: Gunnar Eric Nelson is an American musician.
  Fact 0.2: Gunnar Eric Nelson is an American singer.
  Fact 0.3: Gunnar Eric Nelson is an American songwriter.
  Fact 0.4: Gunnar Eric Nelson is an international multi-platinum recording artist.
Sentence 1: Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the Machine.
  Fact 1.0: Florence Leontine Mary Welch was born on 28 August 1986.
  Fact 1.1: Florence Leontine Mary Welch is an English musician.
  Fact 1.2: Florence Leontine Mary Welch is an English singer.
  Fact 1.3: Florence Leontine Mary Welch is an English songwriter.
  Fact 1.4: Florence Leontine Mary Welch is best known as the vocalist of the indie rock band Florence and the Machine.

Retrieved items:
<item 1>
Gunnar Nelson (musician)
Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
</item 1>

<item 2>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 2>

<item 3>
Lungs (album)
Lungs is the debut studio album by English indie rock band Florence and the Machine, released on 3 July 2009 on Island Records.

 Frontwoman Florence Welch experimented with and honed the album's sound with bandmate Isabella Summers, while also collaborating with five record producers
including James Ford, Paul Epworth, Steve Mackey.

 The music on "Lungs" established Florence and the Machine as an internationally successful rock act, and also featured the group's successful mix of string
arrangements centered around Welch's vocal prowess.
</item 3>

<item 4>
What the Water Gave Me (song)
"What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.

 It was released on 23 August 2011 as the first promotional single from "Ceremonials".

 After the band performed the song in Berkeley, California on 12 June 2011, the studio version premiered on Florence and the Machine's official website.

 Welch decided to name the song "What the Water Gave Me" after seeing a Frida Kahlo painting of the same name.

 During an interview, she confirmed that the song was inspired by English writer Virginia Woolf.
</item 4>

<item 5>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 5>

<item 6>
Sweet Nothing
"Sweet Nothing" is a song by Scottish recording artist and record producer Calvin Harris from his third studio album, "18 Months" (2012).

 It features vocals from Florence and the Machine singer Florence Welch.

 Harris previously worked with the band in a remix of their single, "Spectrum" (2012).

 While recording "18 Months", Harris expressed interest in working with Welch.

 He mentioned that it was not easy and he had to chase her, due to schedule conflicts.

 Welch accepted his invitation and the two recorded "Sweet Nothing".

 The track premiered on "The Chris Moyles Show" on BBC Radio 1 on 28 August 2012 and was later released as the fifth single from the album on 12 October 2012.
</item 6>

<item 7>
Florence Welch (born 1986) leads English indie-rock band Florence + the Machine, formed in 2007 and acclaimed for dramatic production and powerful vocals. Their
debut, *Lungs* (2009), established international success; *Ceremonials* (2011) produced singles including “Shake It Out,” “What the Water Gave Me,” “Spectrum,”
“Never Let Me Go,” “No Light, No Light,” and “Lover to Lover.” Welch
</item 7>

<item 8>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 8>

<item 9>
Florence and the Machine
Florence and the Machine (stylised as Florence + the Machine) are an English indie rock band that formed in London in 2007, consisting of lead singer Florence
Welch, keyboardist Isabella Summers, and a collaboration of other artists.

 The band's music received praise across the media, especially from the BBC, which played a large part in their rise to prominence by promoting Florence and the
Machine as part of "BBC Introducing".

 At the 2009 Brit Awards they received the Brit Awards "Critics' Choice" award.

 The band's music is renowned for its dramatic and eccentric production and also Welch's powerful vocal performances.
</item 9>

<item 10>
Nelson (band)
Nelson is an American rock band founded by singer/songwriters Matthew and Gunnar Nelson (twin sons of Ricky Nelson and Kristin Nelson).

 The band achieved success during the early 1990s with their double platinum debut album "After the Rain", which featured the number-one hit "(Can't Live
Without Your) Love and Affection".
</item 10>
=== END ITEM id=2937f34f656c6584 ===

=== ITEM id=acb614f302c733f1 ===
Gold sentences and facts:
Sentence 0:  It was the first election for Governor of Maine, taking place after Maine separated from Massachusetts and was recognized as a state on March 15, 1820.
  Fact 0.0: The election was the first election for Governor of Maine.
  Fact 0.1: The election took place after Maine separated from Massachusetts.
  Fact 0.2: Maine was recognized as a state on March 15, 1820.
Sentence 1:  Maine's separation from Massachusetts came as a result of The Missouri Compromise.
  Fact 1.0: Maine's separation from Massachusetts came as a result of The Missouri Compromise.
Sentence 2:  This election saw the virtually unanimous election of William King, the man most chiefly responsible for the push for Maine statehood.
  Fact 2.0: The election saw the virtually unanimous election of William King.
  Fact 2.1: William King was the man most chiefly responsible for the push for Maine statehood.
Sentence 3: William King (February 9, 1768June 17, 1852) was an American merchant, shipbuilder, army officer, and statesman from Bath, Maine.
  Fact 3.0: William King was born February 9, 1768.
  Fact 3.1: William King died June 17, 1852.
  Fact 3.2: William King was an American merchant.
  Fact 3.3: William King was an American shipbuilder.
  Fact 3.4: William King was an American army officer.
  Fact 3.5: William King was an American statesman.
  Fact 3.6: William King was from Bath, Maine.

Retrieved items:
<item 1>
Benjamin Logan
Benjamin Logan (c.1742 – December 11, 1802) was an American pioneer, soldier, and politician from Shelby County, Kentucky.

 As colonel of the Kentucky County, Virginia militia during the American Revolutionary War, he was second-in-command of all the militia in Kentucky.

 He was also a leader in Kentucky's efforts to become a state.

 His brother, John Logan, was the first state treasurer of Kentucky.
</item 1>

<item 2>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 2>

<item 3>
Henry Smith Lane
Henry Smith Lane (February 24, 1811 – June 19, 1881) was a United States Representative, Senator, and the 13th Governor of Indiana; he was by design the
shortest-serving Governor of Indiana, having made plans to resign the office should his party take control of the Indiana General Assembly and elect him to the
United States Senate.

 He held that office for only two days, and was known for his opposition to slavery.

 A Whig until the party collapsed, he supported compromise with the south.

 He became an early leader in the Republican Party starting in 1856 serving as the president of the first party convention, delivering its keynote address, and
was influential in the nomination of Abraham Lincoln.

 With the repeal of the Missouri Compromise, he became a full-fledged
</item 3>

<item 4>
John Bowman (pioneer)
Col. John (Johannes) Bowman (17 December 1738 – May 4, 1784) was an 18th-century American pioneer, colonial militia officer and sheriff, the first appointed in
Lincoln County, Kentucky.

 In 1781 he also presided as a justice of the peace over the first county court held in Kentucky.

 The first county-lieutenant and military governor of Kentucky County during the American Revolutionary War, Col. Bowman also, served in the American
Revolution, many times, second in command to General George Rogers Clark, during the Illinois Campaign, which, at the time, doubled the size of the United
States.
</item 4>

<item 5>
Missouri Compromise
The Missouri Compromise is the title generally attached to the legislation passed by the 16th United States Congress on May 8, 1820.

 The measures provided for the admission of Maine as a state along with Missouri as a slave state, thus maintaining the balance of power between North and
South.

 As part of the compromise, slavery was prohibited North of the 36°30′ parallel, excluding Missouri.

 President James Monroe signed the legislation on April 6, 1820.
</item 5>

<item 6>
Anti-Nebraska movement
The Anti-Nebraska movement was a political alignment in the United States formed in opposition to the Kansas-Nebraska Act of 1854 and to its repeal of the
Missouri Compromise provision forbidding slavery in U.S. territories north of latitude 36° 30' N.

 (At the time, the name "Nebraska" could loosely refer to areas west of the Missouri River.)
</item 6>

<item 7>
John Biddle (Michigan politician)
John Biddle (March 2, 1792 – August 25, 1859) was a delegate to the United States Congress from the Michigan Territory.
</item 7>

<item 8>
Compromise of 1790
The Compromise of 1790 was a compromise between Alexander Hamilton on the one hand and Thomas Jefferson and James Madison whereby Hamilton won the decision for
the national government to take over and pay the state debts, while Jefferson and Madison obtained the national capital (District of Columbia) for the South.

 The compromise resolved the deadlock in Congress.

 Southerners were blocking the assumption of state debts by the treasury, thereby destroying the Hamiltonian program for building a fiscally strong nation
state.

 Northerners rejected the proposal, much desired by Virginians, to locate the permanent national capital on the Virginia-Maryland border.

 The compromise made possible the passage of the Residence and Funding (Assumption) Acts in July and August 1790.

 Historian Jacob Cooke says it is, "generally regarded as one of the most important bargains in American history, ranking just below the better known Missouri
Compromise and the Compromise of 1850."
</item 8>

<item 9>
Maine gubernatorial election, 1820
The 1820 Maine gubernatorial election took place on April 3, 1820.

 It was the first election for Governor of Maine, taking place after Maine separated from Massachusetts and was recognized as a state on March 15, 1820.

 Maine's separation from Massachusetts came as a result of The Missouri Compromise.

 This election saw the virtually unanimous election of William King, the man most chiefly responsible for the push for Maine statehood.

 He had no opponents.
</item 9>

<item 10>
William King (governor)
William King (February 9, 1768June 17, 1852) was an American merchant, shipbuilder, army officer, and statesman from Bath, Maine.

 A proponent of statehood for Maine, he became its first governor when it separated from Massachusetts in 1820.
</item 10>
=== END ITEM id=acb614f302c733f1 ===
