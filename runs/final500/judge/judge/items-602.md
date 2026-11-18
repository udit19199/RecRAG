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

=== ITEM id=630997466acc7659 ===
Gold sentences and facts:
Sentence 0:  It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
  Fact 0.0: The work is the third installment of the 'Naked' series.
  Fact 0.1: 'Naked Killer' is a work in the 'Naked' series released in 1992.
  Fact 0.2: 'Naked Weapon' is a work in the 'Naked' series released in 2002.
Sentence 1: Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
  Fact 1.0: Naked Weapon is also titled 赤裸特工.
  Fact 1.1: Naked Weapon is a 2002 film.
  Fact 1.2: Naked Weapon is a Hong Kong film.
  Fact 1.3: Naked Weapon is an action-thriller film.
  Fact 1.4: Naked Weapon was directed by action choreographer Tony Ching.
  Fact 1.5: Naked Weapon stars Maggie Q.
  Fact 1.6: Naked Weapon stars Anya Wu.
  Fact 1.7: Naked Weapon stars Daniel Wu.

Retrieved items:
<item 1>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 1>

<item 2>
Naked Weapon
Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
</item 2>

<item 3>
Anatomy 2
Anatomy 2 (German: "Anatomie 2" ) is a 2003 German thriller film written and directed by Stefan Ruzowitzky.

 It's the sequel to the 2000 film "Anatomy", that starred Franka Potente.

 The story moves to Berlin for this film.
</item 3>

<item 4>
Mel Gibson filmography
Five years later he played Martin Riggs in the buddy cop action comedy "Lethal Weapon" alongside Danny Glovera role he later reprised in its sequels "Lethal
Weapon 2" (1989), "Lethal Weapon 3" (1992), and "Lethal Weapon 4" (1998).
</item 4>

<item 5>
Naked Soldier
Naked Soldier (絕色武器) is a 2012 Hong Kong action film directed by Marco Mak and starring Jennifer Tse.

 It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
</item 5>

<item 6>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 6>

<item 7>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 7>

<item 8>
Naked Gun 33⅓: The Final Insult
Naked Gun 33⅓: The Final Insult is a 1994 comedy film, and the third and final installment in "The Naked Gun" film series, which was based on the "Police Squad!

" television series.
</item 8>

<item 9>
The Naked Gun 2½: The Smell of Fear
The Naked Gun 2½: The Smell of Fear is a 1991 comedy film.

 It is the sequel to the 1988 film "" and the second installment in "The Naked Gun" film series.

 The film stars Leslie Nielsen as the comically bumbling Police Lt. Frank Drebin of "Police Squad!

".

 Priscilla Presley plays the role of Jane, with O. J. Simpson as Nordberg and George Kennedy as police captain Ed Hocken.

 The film also features Robert Goulet (who previously made a "special guest star" appearance on "Police Squad!")

 as the villainous Quentin Hapsburg and Richard Griffiths as renewable fuel advocate Dr. Albert S. Meinheimer (as well as his evil double, Earl Hacker).

 Zsa Zsa Gabor, Mel Tormé and members of the Chicago Bears have cameo roles.
</item 9>

<item 10>
American Pie Presents: The Naked Mile
American Pie Presents: The Naked Mile (also known as American Pie: The Naked Mile) is a 2006 American sex comedy film released by Universal Pictures.

 It is the second installment in the "American Pie Presents" series and the fifth installment in the "American Pie" franchise.

 The film begins a story arc that concludes with "" (2007).

 John White stars as Erik Stifler, a high school senior who is given a "guilt free pass" by his girlfriend, Tracy Sterling (Jessy Schram), and so visits the
Beta House fraternity led by his cousin, Dwight Stifler (Steve Talley), to run a mile naked.

 Christopher McDonald co-stars as Erik's father, Harry, and Eugene Levy once again plays Jim's Dad, who turns out to be a family friend of both Erik's and
Tracy's.

 Also, it is in this film that his name is revealed to be "Noah Levenstein".
</item 10>
=== END ITEM id=630997466acc7659 ===

=== ITEM id=b8980283cf723048 ===
Gold sentences and facts:
Sentence 0: Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.
  Fact 0.0: Adam Fox lived from 1883 to 1977.
  Fact 0.1: Adam Fox was a Canon.
  Fact 0.2: Adam Fox was the Dean of Divinity at Magdalen College, Oxford.
Sentence 1:  He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.
  Fact 1.0: Adam Fox was one of the first members of the Inklings.
  Fact 1.1: The Inklings was a literary group.
  Fact 1.2: The Inklings also included C.S. Lewis.
  Fact 1.3: The Inklings also included J.R.R. Tolkien.
Sentence 2: The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and late 1949.
  Fact 2.0: The Inklings was an informal literary discussion group.
  Fact 2.1: The Inklings was associated with the University of Oxford, England.
  Fact 2.2: The Inklings lasted for nearly two decades.
  Fact 2.3: The Inklings operated between the early 1930s and late 1949.

Retrieved items:
<item 1>
Roger Lancelyn Green
Roger (Gilbert) Lancelyn Green (2 November 1918 – 8 October 1987) was a British biographer and children's writer.

 He was an Oxford academic who formed part of the Inklings literary discussion group along with C. S. Lewis and J. R. R. Tolkien.
</item 1>

<item 2>
Inklings
The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and
late 1949.

 The Inklings were literary enthusiasts who praised the value of narrative in fiction and encouraged the writing of fantasy.
</item 2>

<item 3>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 3>

<item 4>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 4>

<item 5>
Eagle Street College
The Eagle Street College was an informal literary society established in 1885 at the home of James William Wallace in Eagle Street, Bolton, to read and discuss
literary works, particularly the poetry of Walt Whitman, (1819–91).

 The group subsequently became known as the Bolton Whitman Fellowship or Whitmanites.

 Its founder members were Wallace, Dr John Johnston and Fred Wild.

 The group held an annual 'Whitman Day' celebration around the 31 May which was the poet's birthday.
</item 5>

<item 6>
Colin Hardie
Colin Graham Hardie (16 February 1906 – 17 October 1998) was a British classicist and academic.

 From 1933 to 1936, he was Director of the British School at Rome.

 From 1936 to 1973, he was a Fellow of Magdalen College, Oxford, and a tutor in classics.

 In addition, from 1967 to 1973, he was the Public Orator of the University of Oxford.

 He was a member of the Inklings, an informal literary discussion group which included the likes of J. R. R. Tolkien and C. S. Lewis.
</item 6>

<item 7>
Charles Leslie Wrenn
Charles Leslie Wrenn (1895–1969) was a British scholar.

 He became Rawlinson and Bosworth Professor of Anglo-Saxon at the University of Oxford in 1945, the successor in the chair of J.R.R. Tolkien, and held the
position until 1963.

 Wrenn was a Fellow of Pembroke College, Oxford.

 He was also a member of the Oxford literary discussion group known as the "Inklings", which included C. S. Lewis and Tolkien.

 Some of the work published by Wrenn includes "The English Language" (1949), "A Study of Old English Literature" (1967)"," and "An Old English Grammar," written
with Randolph Quirk (1955, rev. 1957).

 His literary interests were primarily comparative literature and later poets including T. S. Elliot.
</item 7>

<item 8>
Tobe Levin
Tobe Levin Freifrau von Gleichen (*Feb 16, 1948), a multi-lingual scholar, translator, editor and activist, is an Associate of the Hutchins Center for African
and African American Research at Harvard University; a Visiting Research Fellow at the International Gender Studies Centre, Lady Margaret Hall, University of
Oxford; an activist against female genital mutilation (FGM) and professor of English Emerita at the University of Maryland, University College.

 Having received her PhD in 1979 from Cornell University, she is most known for combining her advocacy against FGM with her academic scholarship in comparative
literature.

 She has published peer-reviewed and popular articles and book chapters, edited four books, launched UnCUT/VOICES Press in 2009 and founded "Feminist Europa
Review of Books" (1998-2010).

 Her most notable works to date are "Empathy and Rage.

 Female Genital Mutilation in African Literature" and " Waging Empathy.

 Alice Walker, Possessing the Secret of Joy, and the Global Movement to Ban FGM."
</item 8>

<item 9>
James F. Light
James F. Light was an American literary scholar, university vice president, and provost.

 During his academic career, he helped revive the works of satirist Nathanael West, with the first book length critical study of his work, "Nathanael West: An
Interpretive Study", (Northwestern Univ.

 Press, 1961).

 He was also the leading authority on John William De Forest, the early American realist whose work he critiqued in "John William De Forest" (Twayne Pub.,
1965), and he wrote extensively on J. D. Salinger, Robert Penn Warren and others.
</item 9>

<item 10>
John Whenham
John Whenham is an English musicologist and academic who specializes in early Italian baroque music.

 He earned both a Bachelor of Music and a Master of Music from the University of Nottingham, and a Doctor of Philosophy from the University of Oxford.

 He is a leading expert on the life and works of Claudio Monteverdi, and is the author of the books "Duet and Dialogue in the Age of Monteverdi" (Ann Arbor,
Michigan: University Microfilms International, 1982) "Monteverdi, 'Orfeo' " (London: Cambridge University Press, 1986), "Monteverdi, Vespers (1610)" (Cambridge
University Press, 1997), and "The Cambridge Companion to Monteverdi" (with Richard Wistreich, Cambridge University Press, 2007).

 For five years he was co-editor of the journal "Music & Letters".

 He currently serves on the board of the Birmingham Early Music Festival and is head of the music history department at the University of Birmingham.
</item 10>
=== END ITEM id=b8980283cf723048 ===

=== ITEM id=75f9d4dd30eb4b33 ===
Gold sentences and facts:
Sentence 0: "What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).
  Fact 0.0: What the Water Gave Me is a song
  Fact 0.1: What the Water Gave Me is by English indie rock band Florence and the Machine
  Fact 0.2: What the Water Gave Me is from the band's second studio album Ceremonials
  Fact 0.3: The album Ceremonials is dated 2011
Sentence 1:  The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.
  Fact 1.0: Lead singer Florence Welch wrote the song What the Water Gave Me
  Fact 1.1: Francis Eg White wrote the song What the Water Gave Me
  Fact 1.2: Paul Epworth produced the song What the Water Gave Me
Sentence 2: What the Water Gave Me ("Lo que el agua me dio" in Spanish) is an oil painting by Frida Kahlo that was completed in 1938.
  Fact 2.0: What the Water Gave Me is an oil painting by Frida Kahlo
  Fact 2.1: The painting is titled Lo que el agua me dio in Spanish
  Fact 2.2: Frida Kahlo completed the painting in 1938

Retrieved items:
<item 1>
Shake It Out
"Shake It Out" is a song by English indie rock band Florence and the Machine, released as the first official single from their second studio album,
"Ceremonials" (2011).

 It was written by Florence Welch and Paul Epworth, while production was handled by Epworth.

 The song was digitally released in Australia on 14 September 2011, and it was available in the United States on 19 October.

 It had its radio debut on XFM on 14 September 2011 in the United Kingdom.

 Welch revealed that the song was written within an hour and according to her it talked about shaking the regrets and the things that were haunting her.
</item 1>

<item 2>
Florence and the Machine
Florence and the Machine (stylised as Florence + the Machine) are an English indie rock band that formed in London in 2007, consisting of lead singer Florence
Welch, keyboardist Isabella Summers, and a collaboration of other artists.

 The band's music received praise across the media, especially from the BBC, which played a large part in their rise to prominence by promoting Florence and the
Machine as part of "BBC Introducing".

 At the 2009 Brit Awards they received the Brit Awards "Critics' Choice" award.

 The band's music is renowned for its dramatic and eccentric production and also Welch's powerful vocal performances.
</item 2>

<item 3>
What the Water Gave Me (song)
"What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.

 It was released on 23 August 2011 as the first promotional single from "Ceremonials".

 After the band performed the song in Berkeley, California on 12 June 2011, the studio version premiered on Florence and the Machine's official website.

 Welch decided to name the song "What the Water Gave Me" after seeing a Frida Kahlo painting of the same name.

 During an interview, she confirmed that the song was inspired by English writer Virginia Woolf.
</item 3>

<item 4>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 4>

<item 5>
Wish That You Were Here
"Wish That You Were Here" is a song by the English indie rock band Florence and the Machine, written by Florence Welch, Andrew Wyatt, and Emilie Haynie, and was
released on Island Records on 26 August 2016.

 The song was made available via digital download, and is featured on the soundtrack of the film "Miss Peregrine's Home for Peculiar Children" which was
directed by Tim Burton.

 Welch — a long-time fan of Burton's work — had expressed interest in collaborating with him long before the recording of "Wish That You Were Here" as they
shared similar artistic themes.

 Upon release, the composition was positively received by music critics and reached number 128 on the UK Singles Chart.
</item 5>

<item 6>
No Light, No Light
No Light, No Light is a song by English indie rock band Florence and the Machine from their second studio album, "Ceremonials" (2011).

 The song was written by band members Florence Welch and Isabella Summers while the production was handled by Paul Epworth.

 Island Records released the song as the second single from the album on 16 January 2012.

 The song was the first one written for the album in the band's tour bus in Amsterdam.

 Lyrically, in the song, Welch is expressing frustration about the state of her fragile relationship and she further tries to keep it together.

 "No Light, No Light" received positive reviews by music critics who generally praised Welch's vocals and the drum-led instrumentation.

 It was also placed on several critics' year-end lists of best singles.

 The song peaked at number fifty on the UK Singles Chart, the Irish Singles Chart and number thirty nine on the US "Billboard" Alternative Songs chart.
</item 6>

<item 7>
Spectrum (Say My Name)
"Spectrum" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 It was written by lead singer Florence Welch and Paul Epworth with production being handled by Epworth.

 The band premiered the song during a concert at Brooklyn's Creators Project on 15 October 2011, prior to the release of the album.

 The album version of "Spectrum" is a downtempo orchestral pop, R&B, krautrock and disco song.

 The accompanying music video for the song premiered on 30 May 2012.

 It was directed by David LaChapelle and John Byrne.
</item 7>

<item 8>
Never Let Me Go (Florence and the Machine song)
"Never Let Me Go" is a song by English indie rock band Florence and the Machine from their second studio album, "Ceremonials" (2011).

 The song was written by Florence Welch and Paul Epworth while production was handled by Epworth.

 Island Records released the song as the third single from the album on 30 March 2012.

 The band first premiered the song during one of their concerts before the release of the album on 15 October 2011.

 A Clams Casino remix of the song was placed on the 12" white vinyl single which was available for pre-order on 22 March 2012.
</item 8>

<item 9>
Lungs (album)
Lungs is the debut studio album by English indie rock band Florence and the Machine, released on 3 July 2009 on Island Records.

 Frontwoman Florence Welch experimented with and honed the album's sound with bandmate Isabella Summers, while also collaborating with five record producers
including James Ford, Paul Epworth, Steve Mackey.

 The music on "Lungs" established Florence and the Machine as an internationally successful rock act, and also featured the group's successful mix of string
arrangements centered around Welch's vocal prowess.
</item 9>

<item 10>
Lover to Lover
"Lover to Lover" is a song by English indie rock band Florence and the Machine.

 The song was released on 30 November 2012, as a digital download on iTunes as the fifth and final single from their second studio album "Ceremonials" (2011).

 The song was written by Florence Welch and Francis "Eg" White and produced by Paul Epworth.

 The Ceremonials Tour version of "Lover to Lover" is an edited version of the original with further production by Chris Hayden.

 The song became the band's first single to miss the top 200 in the UK Singles Chart.
</item 10>
=== END ITEM id=75f9d4dd30eb4b33 ===

=== ITEM id=62533b36e2e58fc0 ===
Gold sentences and facts:
Sentence 0: San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San Antonio metropolitan area.
  Fact 0.0: San Antonio International Airport has the IATA code SAT.
  Fact 0.1: San Antonio International Airport has the ICAO code KSAT.
  Fact 0.2: San Antonio International Airport has the FAA LID SAT.
  Fact 0.3: San Antonio International Airport is an international airport.
  Fact 0.4: San Antonio International Airport is located in San Antonio, Texas.
  Fact 0.5: San Antonio International Airport serves the Greater San Antonio metropolitan area.
Sentence 1: Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.
  Fact 1.0: Yakutat Airport has the IATA code YAK.
  Fact 1.1: Yakutat Airport has the ICAO code PAYA.
  Fact 1.2: Yakutat Airport has the FAA LID YAK.
  Fact 1.3: Yakutat Airport is a state owned, public use airport.
  Fact 1.4: Yakutat Airport is located three nautical miles (6 km) southeast of the central business district of Yakutat.
  Fact 1.5: Yakutat is a city and borough in the U.S. state of Alaska.
  Fact 1.6: Yakutat has no road access to the outside world.

Retrieved items:
<item 1>
Yakutat Airport
Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business
district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.

 Airline service is subsidized by the Essential Air Service program.
</item 1>

<item 2>
Ta'if Regional Airport
Ta'if Regional Airport (IATA: TIF, ICAO: OETF) is an airport in Ta'if, Saudi Arabia.

 Despite its name, it also offers many international flights as well.

 The airport is located 30 km to the east of Taif and 70 km from Mecca.

 The airport is considered important in Saudi aviation history as it witnessed the first landing of Ibn Saud's plane, founder of the Kingdom.

 It was converted to a regional airport in 2009 when GACA allowed international airlines to operate at the airport as the city's population was increasing and
to reduce pressure on the three main airports at the time.

 Despite being named as a "Regional" airport, the airport actually has international flights destinations in half a dozen countries outside of Saudi Arabia.
</item 2>

<item 3>
Stepanakert Airport
Stepanakert Airport (Armenian: Ստեփանակերտի Օդանավակայան ), is an airport in Khojaly, near Stepanakert, the capital city of the "de facto" independent Republic
of Artsakh (NKR) which is an internationally recognized part of Azerbaijan.

 The airport has been under the control of the Republic of Artsakh since 1992.

 Flights ceased to take place with the escalation of the Nagorno-Karabakh War in 1990.

 As the airport lies within the internationally recognized borders of Azerbaijan, and the current government is unrecognized, there are no codes for the airport
in the official IATA list.
</item 3>

<item 4>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 4>

<item 5>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 5>

<item 6>
Sharp County Regional Airport
Sharp County Regional Airport (IATA: CKK, ICAO: KCVK, FAA LID: CVK) is a public-use airport in Sharp County, Arkansas, United States.

 It is located three nautical miles (4 mi, 6 km) northeast of the central business district of Ash Flat, Arkansas.

 The airport is owned by the Sharp County Regional Airport Authority.
</item 6>

<item 7>
Graph query row:
airport: San Antonio International Airport
locations_and_countries: ['Texas', 'United States', 'North America']
relationships: ['LOCATED_IN', 'LOCATED_IN', 'LOCATED_IN']
</item 7>

<item 8>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 8>

<item 9>
Texarkana Regional Airport
Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km)
northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.

 It is owned by the Texarkana Airport Authority.

 The airport is located within the city limits of Texarkana, roughly 3 miles east of State Line Avenue.

 The front gate opens to the northwest, at the intersection of Arkansas Boulevard and U.S. Route 67.

 A Union Pacific Railroad line runs parallel to US 67 on the side of the highway facing the airport.

 It is mostly used for general aviation, but is also served by American Eagle.
</item 9>

<item 10>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 10>
=== END ITEM id=62533b36e2e58fc0 ===

=== ITEM id=d6b8d0e8977fb77b ===
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
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 2>

<item 3>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
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
Graph query row:
film: The Black Hole
classification: film
genre: science fiction
relationship: None
connected_fact: None
fact_kind: None
</item 5>

<item 6>
Full Metal Jacket
Full Metal Jacket is a 1987 British-American war film directed and produced by Stanley Kubrick.

 The screenplay by Kubrick, Michael Herr, and Gustav Hasford was based on Hasford's novel "The Short-Timers" (1979).

 Its storyline follows a platoon of U.S. Marines through their training, primarily focusing on two privates, Joker and Pyle, who struggle to get through camp
under their foul-mouthed drill instructor, Gunnery Sergeant Hartman, and the experiences of two of the platoon's Marines in the Tet Offensive during the Vietnam
War.

 The film's title refers to the full metal jacket bullet used by soldiers.

 The film was released in the United States on June 26, 1987.
</item 6>

<item 7>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 7>

<item 8>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
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
Graph query row:
film: McFarland, USA
classification: sports drama film
genre: None
relationship: None
connected_fact: None
fact_kind: None
</item 10>
=== END ITEM id=d6b8d0e8977fb77b ===

=== ITEM id=5f571bc53afa0030 ===
Gold sentences and facts:
Sentence 0: Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.
  Fact 0.0: Marc Predka was born April 21, 1975.
  Fact 0.1: Marc Predka is better known by his stage name Tha Trademarc.
  Fact 0.2: Marc Predka is an American hip hop artist.
Sentence 1:  He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance theme "The Time Is Now".
  Fact 1.0: Marc Predka first came to prominence when he collaborated with his first cousin, John Cena.
  Fact 1.1: John Cena is a WWE wrestler.
  Fact 1.2: Marc Predka and John Cena collaborated on the 2005 album 'You Can't See Me'.
  Fact 1.3: The album 'You Can't See Me' includes Cena's entrance theme 'The Time Is Now'.
Sentence 2: "The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.
  Fact 2.0: 'The Time Is Now' is a song by John Cena.
  Fact 2.1: John Cena is a professional wrestler.
  Fact 2.2: John Cena is an actor.
  Fact 2.3: John Cena is a hip hop musician.

Retrieved items:
<item 1>
Robert Glasper
Robert Glasper (born April 6, 1978, in Houston, Texas) is an American pianist and record producer.

 He has been nominated for 6 Grammys, has won 3 Grammy Awards and is currently nominated for an Emmy Award.

 His 2012 album "Black Radio" won the Grammy Award for Best R&B Album at the 55th Grammy Awards.

 His 2014 album "Black Radio 2" won the Grammy Award for Best Traditional R&B Performance at the 56th Grammy Awards.

 The song "These Walls" from Kendrick Lamar's album "To Pimp A Butterfly" won Best Rap/Sung Collaboration at the 57th Grammy Awards, on which Glasper plays
keys.

 The soundtrack for the film "Miles Ahead" won Best Soundtrack Compilation at the 58th Grammy Awards, for which Glasper was a producer.

 The song "Letter To The Free", written with Common, is nominated for an Emmy Award for Best Original Song in the Ava Duvernay documentary film "13th" (Netflix)
at the 2017 Emmys.
</item 1>

<item 2>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 2>

<item 3>
Den Za Nas
Den Za Nas (in Macedonian: Ден за нас ) is the title of the fourth Macedonian language album by the Macedonian superstar Toše Proeski released in 2004.

 It is a 2-CD set which contains all of the songs performed on the National final in order to select a song for the Eurovision Song Contest 2004 with which
Proeski would represent Macedonia in the contest.

 The winning song was "Life" which qualified for the final and finished on the 14th place in the final.

 Some extra songs such as "Dzvezdo Severnice", "Me Sakaš So Zborovi", "Ima Li Den Za Nas" and others were recorded especially for the album.
</item 3>

<item 4>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 4>

<item 5>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 5>

<item 6>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 6>

<item 7>
Tha Trademarc
Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.

 He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance
theme "The Time Is Now".

 He also appeared in the music videos of the songs, "Bad Bad Man" and "Right Now" with Cena.

 Prior to the album, Trademarc appeared on Cena's first WWE DVD "Word Life".

 He now works as a classroom specialist for the Peace Corps of America.
</item 7>

<item 8>
The Time Is Now (John Cena song)
"The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.

 It is the current entrance theme Cena uses as he makes his entrance to the ring.

 Currently wrestling for WWE, Cena recorded the song in 2005 for his debut studio album, "You Can't See Me".
</item 8>

<item 9>
Matt Thiessen
Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist,
pianist, and primary songwriter for the Christian rock band Relient K.

 With Relient K, he has released eight full-length albums, including three that were certified Gold, and three that peaked in the top twenty on the "Billboard"
200.

 Outside of his work with Relient K, Thiessen maintains a low-profile side project called Matthew Thiessen and the Earthquakes.

 In 2009, he co-produced and collaborated on Owl City's album "Ocean Eyes".
</item 9>

<item 10>
Treat (album)
Treat is a split cassette shared between by Dutch punk band The Ex and Scottish ex-pat tour mates Dog Faced Hermans.

 The album was recorded live while the two bands toured Europe together and was released only on cassette in 1990.

 That year the two bands also collaborated on the single "Lied der Steinklopfer" ("Stonestamper's Song") released under the name Ex Faced Hermans, as well as
sharing live sound engineer Gert-Jan, credited as a full member of the Dog Faced Hermans who continued to tour with The Ex for more than a decade.
</item 10>
=== END ITEM id=5f571bc53afa0030 ===

=== ITEM id=003b42695f60dc7a ===
Gold sentences and facts:
Sentence 0:  He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.
  Fact 0.0: The writer has written three novels.
  Fact 0.1: The writer has written three collections of short stories.
  Fact 0.2: The writer has written a collection of novellas.
  Fact 0.3: The writer has written seven volumes of essays.
  Fact 0.4: Three of the writer's essay volumes have won National Book Critics Circle Award prizes.
  Fact 0.5: One of the writer's essay volumes, 'A Temple of Texts' (2006), won the Truman Capote Award for Literary Criticism.
Sentence 1: Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for Literature.
  Fact 1.0: Naguib Mahfouz was born on December 11, 1911.
  Fact 1.1: Naguib Mahfouz died on August 30, 2006.
  Fact 1.2: Naguib Mahfouz's Arabic name is نجيب محفوظ, transliterated Nagīb Maḥfūẓ.
  Fact 1.3: Naguib Mahfouz was an Egyptian writer.
  Fact 1.4: Naguib Mahfouz won the 1988 Nobel Prize for Literature.

Retrieved items:
<item 1>
Frances Liardet
Frances E. Liardet is a writer and translator of Arabic literature.

 She studied Creative Writing at the University of East Anglia.

 She has translated several book-length works, including two books by the modernist Egyptian writer Edwar al-Kharrat and one by Nobel Prize winner Naguib
Mahfouz.
</item 1>

<item 2>
Cairo Trilogy
The Cairo Trilogy (Arabic: الثلاثية‎ ‎ ("The Trilogy") or ثلاثية القاهرة ("The Cairo Trilogy")) is a trilogy of novels written by the Egyptian novelist and
Nobel Prize winner Naguib Mahfouz, and one of the prime works of his literary career.
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
Naguib Mahfouz
Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for
Literature.

 He is regarded as one of the first contemporary writers of Arabic literature, along with Tawfiq el-Hakim, to explore themes of existentialism.

 He published 34 novels, over 350 short stories, dozens of movie scripts, and five plays over a 70-year career.

 Many of his works have been made into Egyptian and foreign films.
</item 4>

<item 5>
Thebes at War
Thebes at War is an early novel by the Egyptian writer Naguib Mahfouz.

 It was originally published in Arabic in 1944.

 An English translation by Humphrey Davies appeared in 2003.

 The novel is one of several that Mahfouz wrote at the beginning of his career, with Pharaonic Egypt as their setting.

 Others in this series of novels include "Khufu's Wisdom" (1939) and "Rhadopis of Nubia" (1943).

 All have been translated into English and appeared in one volume under the title "Three Novels of Ancient Egypt" (Everyman's Library, 2007).
</item 5>

<item 6>
Nobel Prize in Literature
Since 1901, the Nobel Prize in Literature (Swedish: "Nobelpriset i litteratur" ) has been awarded annually to an author from any country who has, in the words
of the will of Alfred Nobel, produced "in the field of literature the most outstanding work in an ideal direction" (original Swedish: "den som inom litteraturen
har producerat det mest framstående verket i en idealisk riktning").

 Though individual works are sometimes cited as being particularly noteworthy, here "work" refers to an author's work as a whole.

 The Swedish Academy decides who, if anyone, will receive the prize in any given year.

 The academy announces the name of the chosen laureate in early October.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895; the others are the Nobel Prize in Chemistry, Nobel Prize in Physics, Nobel
Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 6>

<item 7>
Arabian Nights and Days
Arabian Nights and Days (1979) is a novel by Egyptian writer Naguib Mahfouz, winner of the Nobel Prize for Literature.

 The novel serves as a sequel and companion piece for "One Thousand and One Nights" and includes many of the same characters that appeared in the original work
such as Shahryar, Scheherazade, and Aladdin.
</item 7>

<item 8>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 8>

<item 9>
The Beginning and the End (novel)
The Beginning and the End (بداية ونهاية) is a novel by Naguib Mahfouz, who was awarded the Nobel Prize in literature in 1988.

 An Egyptian, Naguib has been credited with modernizing Arabic literature, with his prolific writing style and his themes on existentialism.

 The novel is marked by very bold characterization by Naguib Mahfouz, for the time period that the story is set in.

 The story moves at a prolific pace and tries to look at the world from each characters view point.
</item 9>

<item 10>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 10>
=== END ITEM id=003b42695f60dc7a ===

=== ITEM id=08f3cdbaba04a72a ===
Gold sentences and facts:
Sentence 0: Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate, representing the 13th District since 1996.
  Fact 0.0: Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate.
  Fact 0.1: Suzanne N.J. 'Susie' Chun Oakland represents the 13th District.
  Fact 0.2: Suzanne N.J. 'Susie' Chun Oakland has represented the 13th District since 1996.
Sentence 1:  Previously she was a member of the Hawaii House of Representatives from 1990-96.
  Fact 1.0: Suzanne N.J. 'Susie' Chun Oakland was previously a member of the Hawaii House of Representatives.
  Fact 1.1: Suzanne N.J. 'Susie' Chun Oakland was a member of the Hawaii House of Representatives from 1990 to 1996.
Sentence 2:  The lower chamber of the legislature is the Hawaiʻi House of Representatives.
  Fact 2.0: The lower chamber of the Hawaii legislature is the Hawaiʻi House of Representatives.

Retrieved items:
<item 1>
Hawaii Senate
The Hawaiʻi State Senate is the upper chamber of the Hawaii State Legislature.

 The senate consists of twenty-five members elected from an equal number of constituent districts across the islands.

 The senate is led by the President of the Senate, elected from the membership of the body, currently Ron Kouchi.

 The forerunner of the Hawaii State Senate during the government of the Kingdom of Hawaiʻ i was the House of Nobles originated in 1840.

 In 1894 the Constitution of the Republic of Hawaii renamed the upper house the present senate.

 Senators are elected to four-year terms and are not subject to term limits.

 Like most state legislatures in the United States, the Hawaii State Senate is a part-time body and senators often have active careers outside government.

 The lower chamber of the legislature is the Hawaiʻi House of Representatives.

 The membership of the Senate also elects additional officers to include the Senate Vice President, Senate Chief Clerk, Assistant Chief Clerk, Senate Sergeant
at Arms and Assistant Sergeant at Arms.
</item 1>

<item 2>
Aloha Dalire
Fabianne Pomaialoha Wong Dalire, known professionally as Aloha Dalire, (June 22, 1950 – August 6, 2014) was an American Hawaiian kumu hula, or master hula
teacher.

 She won the first Miss Aloha Hula as Aloha Wong, in 1971, the same year that the Merrie Monarch Festival was established.

 The Miss Aloha Hula title is hula's top solo wahine (women's) honor.
</item 2>

<item 3>
Hawaii State Capitol
The Hawaii State Capitol is the official statehouse or capitol building of the U.S. state of Hawaii.

 From its chambers, the executive and legislative branches perform the duties involved in governing the state.

 The Hawaii State Legislature—composed of the twenty-five member Hawaii State Senate led by the President of the Senate and the fifty-one member Hawaii State
House of Representatives led by the Speaker of the House—convenes in the building.

 Its principal tenants are the Governor of Hawaii and Lieutenant Governor of Hawaii, as well as all legislative offices and the Legislative Reference Bureau.
</item 3>

<item 4>
Suzanne Chun Oakland
Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate, representing the 13th District since 1996.

 Previously she was a member of the Hawaii House of Representatives from 1990-96.

 She is generally considered to have liberal political views.
</item 4>

<item 5>
Charles Hastings Judd
Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.

 He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
</item 5>

<item 6>
Oregon Territorial Legislature
Oregon’s Territorial Legislature was a bicameral legislative body created by the United States Congress in 1848 as the legislative branch of the government of
the Oregon Territory.

 The upper chamber Council and lower chamber House of Representatives first met in July 1849; they served as the region's legislative body until Oregon became a
state in February 1859, when they were replaced by the bicameral Oregon State Legislature.
</item 6>

<item 7>
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 7>

<item 8>
Alaska Legislature
The Alaska Legislature is the state legislature of the U.S. state of Alaska.

 It is a bicameral institution, consisting of the lower Alaska House of Representatives, with 40 members, and the upper house Alaska Senate, with 20 members.

 There are 40 House Districts (1-40) and 20 Senate Districts (A-T).

 With a total of 60 lawmakers, the Alaska Legislature is the smallest bicameral state legislature in the United States and the second-smallest of all state
legislatures (only the 49-member unicameral Nebraska Legislature is smaller).

 There are no term limits for either chamber.
</item 8>

<item 9>
Hawaii State Legislature
The Hawaii State Legislature is the state legislature of the U.S. state of Hawaii.

 The state legislature is a bicameral body consisting of a lower house, the Hawaii State House of Representatives, with 51 representatives, and an upper house,
the 25-member Hawaii State Senate.

 There are a total of 76 representatives in the legislature, each representing single member districts across the islands.

 The powers of the legislature are granted under Article III of the Constitution of Hawaii.
</item 9>

<item 10>
Maryland General Assembly
The Maryland General Assembly is the state legislature of the U.S. state of Maryland that convenes within the State House in Annapolis.

 It is a bicameral body: the upper chamber, the Maryland State Senate, has 47 representatives and the lower chamber, the Maryland House of Delegates, has 141
representatives.

 Members of both houses serve four-year terms.

 Each house elects its own officers, judges the qualifications and election of its own members, establishes rules for the conduct of its business, and may
punish or expel its own members.
</item 10>
=== END ITEM id=08f3cdbaba04a72a ===

=== ITEM id=998f9b8c09674147 ===
Gold sentences and facts:
Sentence 0: St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.
  Fact 0.0: St. Elizabeths Hospital opened in 1855.
  Fact 0.1: St. Elizabeths Hospital opened as the first federally operated psychiatric hospital in the United States.
Sentence 1:  Founded in 1861, it is the longest continuously operating hospital in the District.
  Fact 1.0: St. Elizabeths Hospital was founded in 1861.
  Fact 1.1: St. Elizabeths Hospital is the longest continuously operating hospital in the District.

Retrieved items:
<item 1>
Montefiore New Rochelle Hospital
Montefiore New Rochelle Hospital (formerly Sound Shore Medical Center) is a community-based, teaching hospital located at 16 Guion Place in the city of New
Rochelle, in Westchester County, New York, and affiliated with the Albert Einstein College of Medicine.

 The hospital opened in 1892 as New Rochelle Hospital.

 On November 6, 2013, Sound Shore was acquired by the Bronx-based Montefiore Medical Center, the University Hospital for the Albert Einstein College of
Medicine, was renamed Montefiore New Rochelle Hospital, and became part of the Montefiore Health System.
</item 1>

<item 2>
History of psychiatric hospitals in Pittsburgh
Pittsburgh, Pennsylvania, United States, is home to a plethora of hospitals, be it for the physically ill or, what was defined as the insane and feeble minded.

 In 1891, the St. Margaret Memorial Hospital opened under John H. Shoenberger.

 In 1908, the Pittsburgh Hospital was run under C. A. Fagan, and on June 1, 1889, the South Side Hospital opened under Anna M. Rindlaub.

 Pittsburgh, commonly known as the "Steel City," also built up a reputation for its abundance of hospitals that had opened up within a fairly close time period
of each other.

 During this time, many new mental hospitals were established due to overcrowding.

 In 1899, a building to house the mentally insane, Mayview State Hospital, was constructed; but it was not long before the complex had a reputation “as a place
of sorrow” due to the lack of proper health care patients received during their stay.
</item 2>

<item 3>
St Mary de Lode Church, Gloucester
St Mary de Lode Church, Archdeacon Street, Gloucester GL1 2QT, is a Church of England church immediately outside the grounds of Gloucester Cathedral.

 It is believed by some to be on the site of the first Christian church in Britain.

 The church is in the Diocese of Gloucester and Grade I listed by English Heritage.

 It has also been known as "St. Mary Before the Gate of St. Peter", "St. Mary Broad Gate" and "St. Mary De Port".
</item 3>

<item 4>
St. Elizabeths Hospital
St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.

 Housing over 8,000 patients at its peak in the 1950s, the hospital at one point had a fully functioning medical-surgical unit, a school of nursing, and
accredited internships and psychiatric residencies.

 Its campus was designated a National Historic Landmark in 1990.
</item 4>

<item 5>
Heritage Hospital
Heritage Hospital is a hospital located in Tarboro, North Carolina.

 It is a part of the University Health Systems of Eastern Carolina (UHSEC).

 Edgecombe General Hospital opened as a county-owned hospital in 1916.

 It succeeded Pittman Hospital, which opened in 1901.

 In 1959, the Hill-Burton Act helped combine Edgecombe General Hospital, with three other facilities.

 Edgecombe County sold the hospital to Hospital Corporation of America (HCA) in 1982.

 HCA opened a 127-bed facility in 1985, named Heritage Hospital.

 UHSEC bought Heritage Hospital in 1998 from HCA.

 The hospitals focus is as a community hospital.

 The hospital has 101 general and 16 rehabilitation hospital beds.

 It has five Shared Inpatient/Ambulatory Surgery, two Endoscopy, and one C-Section operating rooms.
</item 5>

<item 6>
Vassar Brothers Medical Center
In 2002 Vassar Brothers Hospital officially became Vassar Brothers Medical Center.

 In September 2016, VBMC announced a new $500 million expansion which would almost double the number of available rooms.

 Construction on the expansion is expected to be completed in 2019.
</item 6>

<item 7>
Vassar Brothers Medical Center
Vassar Brother Medical Center (locally known as Vassar Hospital or VBMC) is a major medical facility located in the city of Poughkeepsie, New York that is a
member of the Health Quest network, a nonprofit family of hospitals and healthcare centers in the Hudson Valley area.

 VBMC is one of two major medical centers located in Dutchess County, New York, the other being Mid-Hudson Regional Hospital of Westchester Medical Center.

 The hospital was first incorporated in 1882 under the name Vassar Brothers Hospital, following contributions made by John Guy Vassar following the death of his
brother.

 The hospital opened its doors in 1887, and initially had 40 beds divided up into four wards, each containing ten beds.

 At the time the hospital opened it also contained a labor and delivery ward, a children's ward, a nursery, three private rooms, and two isolation rooms.

 Several major renovations occurred in 1983, 1991, and 2001; adding critical care areas, a new operating and delivery wing, and a cancer care center,
respectively.
</item 7>

<item 8>
Providence Hospital (Washington, D.C.)
Providence Hospital is a 408 bed hospital located in the District of Columbia.

 Founded in 1861, it is the longest continuously operating hospital in the District.

 Providence Hospital is a member of Ascension Health, the largest non-profit health care organization in the United States.
</item 8>

<item 9>
Westminster St Margaret and St John
St Margaret was an ancient parish in the City and Liberty of Westminster and the county of Middlesex.

 It included the core of modern Westminster, including the Palace of Westminster and the area around, but not including Westminster Abbey.

 In 1727 it was divided into St Margaret's and St John's, to coincide with the building of the Church of St John the Evangelist, constructed by the Commission
for Building Fifty New Churches in Smith Square to meet the demands of the growing population, but there continued to be a single vestry for the parishes of St
Margaret and St John.

 This was reformed in 1855 by the Metropolis Management Act, and the two parishes formed the Westminster District until 1887.

 In 1889 St Margaret and St John became part of the County of London.

 The vestry was abolished in 1900, to be replaced by Westminster City Council, but St Margaret and St John continued to have a nominal existence until 1922.
</item 9>

<item 10>
St. John's Regional Medical Center (California)
St. John's Regional Medical Center is a hospital located in Oxnard, California in the United States, and is operated by Dignity Health, along with its sister
hospital, St. John's Pleasant Valley Hospital in Camarillo, California.

 The hospital was founded in 1912.

 St. John's Regional Medical Center and St. John's Pleasant Valley Hospital offer comprehensive medical services, including 24-hour emergency medical and
surgical services and care, cancer and oncology care and support, cardiovascular care, community outreach and screenings, diagnostic imaging services,
laboratory services, maternity and women's services, neonatal intensive care, palliative care, patient and family education, rehabilitation services, spine and
orthopedic care, weight loss surgery, wound healing and oxygen therapy, and more.

 Together, St. John's Regional Medical Center and St. John's Pleasant Valley Hospital represent the largest acute-care health organization in Ventura County.

 St. John's hospitals serve all of Ventura County and beyond, including the cities of Camarillo, Moorpark, Oxnard, Port Hueneme, Ventura, and Somis.
</item 10>
=== END ITEM id=998f9b8c09674147 ===

=== ITEM id=41ddf43947f03b29 ===
Gold sentences and facts:
Sentence 0: Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).
  Fact 0.0: Groom Lake is a salt flat in Nevada.
  Fact 0.1: Groom Lake is used for runways of the Nellis Bombing Range Test Site airport.
  Fact 0.2: The Nellis Bombing Range Test Site airport has the code KXTA.
Sentence 1:  Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
  Fact 1.0: Groom Lake is located within the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.1: Groom Lake Valley is the namesake of the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.2: Groom Lake is 25 miles south of Rachel, Nevada.
Sentence 2: Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.
  Fact 2.0: Rachel is a census-designated place (CDP).
  Fact 2.1: Rachel is in Lincoln County, Nevada, United States.

Retrieved items:
<item 1>
Papoose Lake
Papoose Lake is a dry lake bed located in Lincoln County, Nevada, USA.

 The lake lies within the plot of land referred to as the Groom Lake facility (aka Area 51) and is a restricted area.

 The lake is located a few miles southwest of the Groom lake facility nestled adjacent the Papoose Range.
</item 1>

<item 2>
Leviathan Mine
Leviathan Mine is a United States superfund site (CERCLIS ID: CAD98067685) at an abandoned open-pit sulfur mine located in Alpine County, California.

 The mine is located at on the eastern slope of the Sierra Nevada at about 7,000 ft elevation, 6 mi east of Markleeville and 24 mi southeast of Lake Tahoe.

 The mine site comprises approximately 250 acre of land surrounded by the Toiyabe National Forest, which is only accessible a few months a year.

 The approximately 22 million tons of sulfur ore-containing crushed rock at the mine are responsible for contaminating the Leviathan and Aspen Creek, which join
with Mountaineer Creek to form Bryant Creek which ultimately empties into the East Fork of the Carson River.

 These water bodies are listed as 303(d) impaired.

 The site location is seismically active.
</item 2>

<item 3>
John Skillern House
The John Skillern House is a historic cabin located 25 mi northwest of Fairfield in Camas County, Idaho, near the confluence of the Big Smokey and Little Smokey
creeks.

 The cabin was built in 1921-22 for John Skillern and his wife, who used it as a summer home and headquarters for John's large sheep ranching business.

 Skillern's wife based the cabin's rustic design off of the Old Faithful Inn in Yellowstone National Park, which is reflected in the cabin's steep roof and
covered front porch.

 The cabin's other significant rustic features include its sleeping lofts with pole railings, its horizontal log construction with exposed logs on the inner
walls, and its stone chimney.
</item 3>

<item 4>
Tonopah Test Range
Tonopah Test Range (TTR), is a restricted military installation located about 30 mi southeast of Tonopah, Nevada.

 It is part of the northern fringe of the Nellis Range, measuring 625 sqmi .

 Tonopah Test Range is located about 70 mi northwest of Groom Dry Lake, the home of the Area 51 facility.

 Like the Groom Lake facility, Tonopah is a site of interest to conspiracy theorists, mostly for its use of experimental and classified aircraft.

 As such, it is not the focus of alien enthusiasts, unlike its neighbor.

 It is currently used for nuclear weapons stockpile reliability testing, research and development of fusing and firing systems, and testing nuclear weapon
delivery systems.

 The airspace comprises restricted area R-4809 of the Nevada Test and Training Range and is often used for military training.
</item 4>

<item 5>
Register Cliff, Wyoming, was an Oregon Trail landmark where emigrants carved names; roughly 500,000 used western trails from 1843–1869. Lafayette Regional
Airport serves Lafayette, Louisiana. Northern California TRACON manages regional air traffic, including major Bay Area and Sacramento airports. Blue Origin
tests New Shepard at Corn Ranch, Texas. Nevada’s Groom Lake is a dry lake and Area 51 runway site; nearby Rachel is known for UFO tourism, and Groom Range rises
to
</item 5>

<item 6>
Rachel, Nevada
Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.

 As of the 2010 census it had a population of 54.

 As the closest habitation to the Nellis Air Force Range and Area 51, Rachel enjoys a modest celebrity, particularly among aviation enthusiasts and UFO hunters.

 North of the town is the Quinn Canyon Range, which has the ghost town of Adaven.
</item 6>

<item 7>
Groom Range
The Groom Range is a mountain range in Lincoln County, Nevada.

 It is located within the Nevada Test and Training Range, north of Groom Dry Lake.

 The highest point in the Groom Range is 9,249 feet.

 The Groom Range is situated 26.6 miles north of the dry Groom Lake.
</item 7>

<item 8>
Saskatchewan Highway 35
Highway 35 is a paved undivided provincial highway in the Canadian province of Saskatchewan.

 It runs from the US Border near Port of Oungre (where it meets United States Route 85) to a dead end near the north shore of Tobin Lake.

 Saskatchewan Highway 35 is about 540 km long.

 The CanAm Highway comprises Saskatchewan Highways 35, SK Hwy 39, SK Hwy 6, SK Hwy 3, SK Hwy 2 and U.S. Route 85.

 46.5 mi of SK Hwy 35 contribute to the CanAm Highway between Port of Oungre on the Canada – United States border and Weyburn.

 Mudslides, and spring flooding were huge road building and maintenance problems around Nipawin as well as along the southern portion of the route named the
"Greater Yellow Grass Marsh".

 Over 20 early dams were built until the problem was addressed with the Rafferty-Alameda Project on the Souris River and the construction of the Qu'Appelle
River Dam which have helped to eliminate washed out roads and flooded communities.
</item 8>

<item 9>
Crystal Lake (Gilmanton, New Hampshire)
Crystal Lake is a 455 acre water body located in Belknap County in the Lakes Region of central New Hampshire, United States, in the town of Gilmanton.

 Crystal Lake is at the head of the Suncook River watershed.

 Water flows into Crystal Lake from the Belknap Range through Manning and Sunset lakes to the north.

 Water from Manning Lake enters Crystal Lake via Nelson Brook.

 Other waterways flowing into Crystal Lake include Wasson Brook and Mill Brook.

 Water flows out of a dam from the southern end of Crystal Lake via the Suncook Lakes and Suncook River to the Merrimack River.

 Shoreline development along Crystal Lake consists primarily of summer cottages, with a few year-round residents.

 Belknap Mountain and Mount Major can be seen from the shoreline.
</item 9>

<item 10>
Groom Lake (salt flat)
Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).

 It is located immediately north of the Area 51 USAF installation.

 The lake at 4409 ft elevation is approximately 3.7 mi from north to south and 3 mi from east to west at its widest point, and is approximately 11.3 miles in
circumference.

 Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
</item 10>
=== END ITEM id=41ddf43947f03b29 ===
