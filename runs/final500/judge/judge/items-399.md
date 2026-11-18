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

=== ITEM id=08ea5b1fd3df7dc5 ===
Gold sentences and facts:
Sentence 0: 'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.
  Fact 0.0: A Guarani word, written in the source as 'aba], means 'what gives strength to the Indian'.
  Fact 0.1: The Guarani word is used for the infusions of the bark of a number of trees native to Brazil.
Sentence 1: Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South America that belongs to the Tupi–Guarani family of the Tupian languages.
  Fact 1.0: Guarani specifically refers to the primary variety known as Paraguayan Guarani.
  Fact 1.1: Paraguayan Guarani has the endonym avañe'ẽ, meaning 'the people's language'.
  Fact 1.2: Guarani is an indigenous language of South America.
  Fact 1.3: Guarani belongs to the Tupi–Guarani family of the Tupian languages.

Retrieved items:
<item 1>
San Antonio River (South America)
The San Antonio River (Spanish, Río San Antonio, Portuguese, Rio Santo Antônio; also called San Antonio Guazú) is a tributary of the Iguazu River.

 The San Antonio River forms the border between Misiones Province in Argentina and Paraná State in Brazil.

 South of the San Antonio's source near Barracão, the international border continues south along the Pepiri-Guazu River, which forms the border between Misiones
Province and Santa Catarina State.
</item 1>

<item 2>
Catuaba
The name catuaba (pronounced [ka.two.

'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.

 The most widely used barks are derived from the trees "Trichilia catigua" and "Erythroxylum vaccinifolium".

 Other catuaba preparations use the bark of trees from the following genera or families: "Anemopaegma", "Ilex", "Micropholis", "Phyllanthus", "Secondatia",
"Tetragastris" and species from the Myrtaceae.
</item 2>

<item 3>
Achomawi
In anthropology, Achomawi (also Achumawi, Ajumawi and Ahjumawi) are the northerly nine (out of eleven) bands of the Pit River tribe of Native Americans who
lived in what is now northeastern California in the United States.

 These nine autonomous bands (also called "tribelets") of the Pit River Indians spoke various dialects of one common language, and the other two bands spoke
dialects of a related language, called Atsugewi.

 The name "Achomawi" means river dwelling (from ajumma, "river") and the band historically inhabited the Fall River Valley and the Pit River from the south end
of Big Valley Mountains, westerly to Pit River Falls.
</item 3>

<item 4>
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 4>

<item 5>
Cora language
Cora is an indigenous language of Mexico of the Uto-Aztecan language family.

 It is spoken by the ethnic group that is widely known as the Cora but who refer to themselves as "Naáyarite".

 The Cora inhabit the northern sierra of the Mexican state Nayarit which is named after its indigenous inhabitants.

 Cora is a Mesoamerican language and shows many of the traits defining the Mesoamerican Linguistic Area.

 Under the General Law of Linguistic Rights of the Indigenous Peoples it is recognized as a "national language" along with 62 other indigenous languages and
Spanish which have the same "validity" in Mexico .
</item 5>

<item 6>
Pagode (music)
Pagode is a type of Brazilian country-folk traditional style of music.

 It's also known as Cipó Preto (black liana), Pagode caipira (rural pagode) or Pagode sertanejo (folk pagode).

 Sertanejo means anything that comes from the back-country, outback or simply the countryside.

 This style of music was pioneered by musicians Tião Carreiro and Lourival dos Santos in the late 1950s, when they fused the "Coco" and "Calango de roda"
rhythms.

 Although other instruments can be added, it is much characterized by its simple, acoustic approach, with the use of just one or two viola guitars, often
accompanied by an acoustic guitar and, at times, a light percussion section.

 A structural feature commonly used in this genre is the alternation of verses sung with the accompaniement of a classical guitar, with solos of viola caipira.

 Pieces can be entirely instrumental or have one or two singers, usually: when there are two lead vocals (that can be men, women or a combination), they sing
together on a steady interval.
</item 6>

<item 7>
Cabécar language
The Cabécar language is an indigenous American language of the Chibchan language family which is spoken by the Cabécar people in Costa Rica.

 Specifically, it is spoken in the inland Turrialba Region of the Cartago Province.

 80% of speakers are monolingual; as of 2007, it is the only indigenous language in Costa Rica with monolingual adults.

 The language is also known by its dialect names Chirripó, Estrella, Telire, and Ujarrás.
</item 7>

<item 8>
Guarani language
Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South
America that belongs to the Tupi–Guarani family of the Tupian languages.

 It is one of the official languages of Paraguay (along with Spanish), where it is spoken by the majority of the population, and where half of the rural
population is monolingual.

 It is spoken by communities in neighboring countries, including parts of northeastern Argentina, southeastern Bolivia and southwestern Brazil, and is a second
official language of the Argentine province of Corrientes since 2004; it is also an official language of Mercosur.
</item 8>

<item 9>
Indigenous language
An indigenous language or autochthonous language is a language that is native to a region and spoken by indigenous people, often reduced to the status of a
minority language.

 This language would be from a linguistically distinct community that has been settled in the area for many generations.

 Indigenous languages are not necessarily national languages, and the reverse is also true.
</item 9>

<item 10>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
</item 10>
=== END ITEM id=08ea5b1fd3df7dc5 ===

=== ITEM id=c5e2f27c29a83058 ===
Gold sentences and facts:
Sentence 0: Robert Harry Philibosian (born 1940) is an American politician.
  Fact 0.0: Robert Harry Philibosian was born in 1940.
  Fact 0.1: Robert Harry Philibosian is an American politician.
Sentence 1: David R. Ignatius (May 26, 1950), is an American journalist and novelist.
  Fact 1.0: David R. Ignatius was born on May 26, 1950.
  Fact 1.1: David R. Ignatius is an American journalist.
  Fact 1.2: David R. Ignatius is an American novelist.

Retrieved items:
<item 1>
Body of Lies (novel)
Body of Lies is an American spy thriller novel by David Ignatius, a columnist for "The Washington Post".

 It was published by W. W. Norton in 2007.

 It was originally titled "Penetration" but was renamed after Warner Bros. bought the rights in 2006.

 It was made into a film of the same name in 2008.
</item 1>

<item 2>
George McKelvey (mayor)
George M. McKelvey is an American politician.

 A Democrat, he served two terms as Mayor of Youngstown, Ohio, from 1998 to 2005.

 Prior to serving as mayor, he served two terms as the Treasurer of Mahoning County.

 He has also been a teacher, school administrator and city council member in Youngstown.
</item 2>

<item 3>
Robert Philibosian
Robert Harry Philibosian (born 1940) is an American politician.

 He served as Los Angeles County District Attorney from 1982 to 1984.

 He received his B.A. in history from Stanford University and his law degree from Southwestern Law School, and was admitted to the California State Bar in 1968.

 He is now Of counsel at the law firm of Sheppard, Mullin, Richter & Hampton.
</item 3>

<item 4>
Charles C. W. Cooke
Charles C. W. Cooke (born November 4, 1984) is the editor of "National Review Online".

 He took the role over after Rich Lowry stepped down in June, 2016 (Lowry remains the editor-in-chief of "National Review").

 Cooke is the author of "The Conservatarian Manifesto" and a frequent guest on HBO's "Real Time with Bill Maher".

 In addition to "National Review", he has written for the "New York Times", the "Washington Post", and the "Los Angeles Times".

 Along with Kevin D. Williamson, he hosts the popular "Mad Dogs and Englishmen" podcast.
</item 4>

<item 5>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 5>

<item 6>
Lois Romano
Lois Romano is an American national journalist who was an editor, reporter and columnist for The Washington Post.

 She was most recently the editor of Washington Post Live, the news organization's editorial events business.

 In early 2015, she returned to the newspaper where she had a long career as a political correspondent and profile writer.

 During her tenure at PostLive, she raised the profile of the platform by linking it to the outlet's award-winning enterprise journalism and utilizing its deep
bench of seasoned journalists as moderators.

 She also created the high value editorial content, such as the successful “Pre-Game” series before presidential debates, a news-driven morning event called
Coffee@WaPo, and the Security Tomorrow conversations with columnist David Ignatius.

 Working with engineers, Romano helped develop the platform's first App.

 Working with newsroom leadership in 2016, Romano also ensured that The Post had its first-ever political convention venues outside the official workspaces in
both Cleveland and Philadelphia.
</item 6>

<item 7>
David Smick
David M. Smick is a macroeconomic adviser to a select group of prominent global investment funds and a non-fiction author.

 He is the chairman and CEO of Johnson Smick International, a financial advisory firm in Washington, D.C. where he is in partnership with former Federal Reserve
Vice Chairman Manuel H. Johnson.

 Smick wrote an acclaimed book "The World Is Curved" on the financial perils of globalization, in response to "The World Is Flat" bestseller by Thomas Friedman.

 The book received positive reviews by the "USA Today" and "Forbes", was called "astonishingly prescient" by David Brooks and "clairvoyant" by David Ignatius.
</item 7>

<item 8>
George H. W. Bush
George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice
President of the United States from 1981 to 1989.

 A member of the Republican Party, he was previously a congressman, ambassador, and Director of Central Intelligence.

 He is the oldest living former President and Vice President.

 Previously known as simply "George Bush", since 2001, Bush has often been referred to as "George H. W. Bush", "Bush 41", "Bush the Elder", or "George Bush
Senior" to distinguish him from his eldest son, George W. Bush, the 43rd President of the United States.
</item 8>

<item 9>
Harold Ford Jr.
Harold Eugene Ford Jr. (born May 11, 1970) is an American politician and author who served five terms in the United States House of Representatives as a member
of the Democratic Party from  's 9 congressional district , centered in Memphis, from 1997 to 2007.

 He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.

 In 2006, Ford made an unsuccessful bid for the US Senate seat vacated by retiring Bill Frist.

 He is a member of the Ford political family from Memphis.

 Ford was the last chairman of the Democratic Leadership Council (DLC).
</item 9>

<item 10>
Enemy Combatant (book)
Enemy Combatant is a memoir by British Muslim, Moazzam Begg, co-written by Victoria Brittain, former Associate Foreign Editor for "The Guardian", about Begg's
detention by the government of the United States of America in Bagram Detention Facility and at Camp Echo, Guantanamo Bay and his life prior to that detention.

 It was published in Britain as "Enemy Combatant: A British Muslim's Journey To Guantanamo and Back" (ISBN  ), and in the US as "Enemy Combatant: My
Imprisonment at Guantanamo, Bagram, and Kandahar" (ISBN  ).

 In the US, the foreword was written by David Ignatius of The Washington Post.
</item 10>
=== END ITEM id=c5e2f27c29a83058 ===

=== ITEM id=1fdfdf22d1e8b82a ===
Gold sentences and facts:
Sentence 0: The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of its kind in Northern Europe.
  Fact 0.0: The Cathedral and Metropolitical Church of Saint Peter in York is commonly known as York Minster.
  Fact 0.1: York Minster is the cathedral of York, England.
  Fact 0.2: York Minster is one of the largest cathedrals of its kind in Northern Europe.
Sentence 1: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.
  Fact 1.0: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

Retrieved items:
<item 1>
Diocese of Salisbury
The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.

 The diocese covers most of Dorset (excepting the deaneries of Bournemouth and Christchurch, which fall within the Diocese of Winchester), and most of Wiltshire
(excepting a part in the north and Swindon).

 The diocese is led by the Bishop of Salisbury (Nick Holtam) and the diocesan synod.

 The bishop's seat is at Salisbury Cathedral.
</item 1>

<item 2>
Minster (church)
Minster is an honorific title given to particular churches in England, most famously York Minster in York, Westminster in London and Southwell Minster in
Southwell.

 The term "minster" is first found in royal foundation charters of the 7th century.

 Although it corresponds to the Latin "monasterium" or monastery, it then designated any settlement of clergy living a communal life and endowed by charter with
the obligation of maintaining the daily office of prayer.

 Widespread in 10th-century Anglo-Saxon England, minsters declined in importance with the systematic introduction of parishes and parish churches from the 11th
century onwards.

 It continued as a title of dignity in later medieval England, for instances where a cathedral, monastery, collegiate church or parish church had originated
with an Anglo-Saxon foundation.

 Eventually a minster came to refer more generally to "any large or important church, especially a collegiate or cathedral church".

 In the 21st century, the Church of England has designated additional minsters by bestowing the status on existing parish churches.
</item 2>

<item 3>
Lincoln Cathedral
Lincoln Cathedral or the Cathedral Church of the Blessed Virgin Mary of Lincoln, and sometimes St. Mary's Cathedral in Lincoln, England is the seat of the
Anglican bishop.

 Building commenced in 1088 and continued in several phases throughout the medieval period.

 It was the tallest building in the world for 238 years (1311–1549), and the first building to hold that title after the Great Pyramid of Giza.

 The central spire collapsed in 1549 and was not rebuilt.

 The cathedral is the third largest in Britain (in floor area) after St Paul's and York Minster, being 484 by .

 It is highly regarded by architectural scholars; the eminent Victorian writer John Ruskin declared: "I have always held... that the cathedral of Lincoln is out
and out the most precious piece of architecture in the British Isles and roughly speaking worth any two other cathedrals we have."
</item 3>

<item 4>
York Minster Police
York Minster Police is a small, specialised cathedral constabulary responsible for policing York Minster in York, United Kingdom.
</item 4>

<item 5>
Old Palace (York)
The Old Palace in the city of York, North Yorkshire, England, is also known as the Minster Library and is in Dean's Park.

 It houses York Minster’s library and archives as well as the Collections Department and conservation studio.

 Its name is a new one and renders homage to the part of the building that used to be the chapel of the Archbishop of York, which was built in the 13th century.

 It was refurbished in 1810 and shortly thereafter became the home of the Minster library.

 Notable items held in the collection include cathedral records dating to back to 1150 and a copy of the 1631 Wicked Bible.

 It is a Grade I listed building.

 An extension was added in 1998.
</item 5>

<item 6>
Brentwood Cathedral
The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

 It is the seat of the Diocese of Brentwood.
</item 6>

<item 7>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 7>

<item 8>
York Minster astronomical clock
The York Minster astronomical clock was installed in the North Transept of York Minster in 1955.

 It was first conceived in 1944 and designed by R d'E Atkinson, chief assistant at the Royal Greenwich Observatory.

 The clock is a memorial to the airmen operating from bases in Yorkshire, Durham, and Northumberland who were killed in action during World War II.
</item 8>

<item 9>
York Minster
The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of
its kind in Northern Europe.

 The minster is the seat of the Archbishop of York, the second-highest office of the Church of England, and is the mother church for the Diocese of York and the
Province of York.

 It is run by a dean and chapter, under the Dean of York.

 The title "minster" is attributed to churches established in the Anglo-Saxon period as missionary teaching churches, and serves now as an honorific title.

 Services in the minster are sometimes regarded as on the High Church or Anglo-Catholic end of the Anglican continuum.
</item 9>

<item 10>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
</item 10>
=== END ITEM id=1fdfdf22d1e8b82a ===

=== ITEM id=b3ec2f53aa7f8d3a ===
Gold sentences and facts:
Sentence 0: Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.
  Fact 0.0: Triumph of the Will's German title is Triumph des Willens.
  Fact 0.1: Triumph of the Will is a 1935 German propaganda film.
  Fact 0.2: Leni Riefenstahl directed Triumph of the Will.
  Fact 0.3: Leni Riefenstahl produced Triumph of the Will.
  Fact 0.4: Leni Riefenstahl edited Triumph of the Will.
  Fact 0.5: Leni Riefenstahl co-wrote Triumph of the Will.
Sentence 1: The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".
  Fact 1.0: The End of America is a 2008 documentary film.
  Fact 1.1: Annie Sundberg directed The End of America.
  Fact 1.2: Ricki Stern directed The End of America.
  Fact 1.3: The End of America is adapted from Naomi Wolf's 2007 book.

Retrieved items:
<item 1>
Olympia (1938 film)
Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium
in Berlin, Germany.

 The film was released in two parts: "Olympia 1.

 Teil — Fest der Völker" (Festival of Nations) and "Olympia 2.

 Teil — Fest der Schönheit" (Festival of Beauty).

 It was the first documentary feature film of the Olympic Games ever made.

 Many advanced motion picture techniques, which later became industry standards but which were groundbreaking at the time, were employed —including unusual
camera angles, smash cuts, extreme close-ups, placing tracking shot rails within the bleachers, and the like.

 The techniques employed are almost universally admired, but the film is controversial due to its political context.

 Nevertheless, the film appears on many lists of the greatest films of all time, including "Time" magazine's "All-Time 100 Movies."
</item 1>

<item 2>
Empire (1964 film)
Empire is a 1964 American black and white silent film written, produced, and directed by Andy Warhol.

 It consists of eight hours and five minutes of continuous slow motion footage of the Empire State Building in New York City.

 Abridged showings of the film were never allowed, and supposedly the unwatchability of the film was an important part of the reason the film was created.

 However, a legitimate Italian VHS produced in association with the Andy Warhol Museum in 2000 contains only an extract of 60 minutes.

 Its use of the long take "in extremis" is an extension of Warhol's earlier work the previous year with "Sleep".

 Warhol employed Rob Trains to be the projectionist for a screening of the film.

 Trains miscalculated and mixed the order and speed of the reels for the eight-hour movie.

 After a positive review in "The New York Times", Warhol actually liked the "mistake" and employed Trains for the entire summer.
</item 2>

<item 3>
Love Begins at 20
Love Begins at 20 is a 1936 American comedy film directed by Frank McDonald and written by Dalton Trumbo and Tom Reed, based on the 1929 play "Broken Dishes" by
Martin Flavin.

 The film stars Hugh Herbert, Patricia Ellis, Warren Hull, Hobart Cavanaugh, Dorothy Vaughan and Clarence Wilson.

 The film was released by Warner Bros. on August 22, 1936.
</item 3>

<item 4>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 4>

<item 5>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 5>

<item 6>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 6>

<item 7>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 7>

<item 8>
The End of America (film)
The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".

 The film analyzes policy changes made during the Bush Administration, and makes the case that these changes threaten American democracy.
</item 8>

<item 9>
Apocalypse Now
Apocalypse Now is a 1979 American epic war film directed, produced, and co-written by Francis Ford Coppola.

 It was co-written by John Milius with narration written by Michael Herr.

 It stars Marlon Brando, Robert Duvall, Martin Sheen, Frederic Forrest, Albert Hall, Sam Bottoms, Larry Fishburne, and Dennis Hopper.

 The screenplay, written by Milius, adapts the story of Joseph Conrad's novella "Heart of Darkness", changing its setting from late nineteenth-century Congo to
the Vietnam War.

 It draws from Herr's "Dispatches" and Werner Herzog's "Aguirre, the Wrath of God" (1972).

 The film revolves around Captain Benjamin L. Willard (Sheen) on a secret mission to assassinate Colonel Kurtz, a renegade who is presumed insane.
</item 9>

<item 10>
Triumph of the Will
Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.

 It chronicles the 1934 Nazi Party Congress in Nuremberg, which was attended by more than 700,000 Nazi supporters.

 The film contains excerpts from speeches given by Nazi leaders at the Congress, including Adolf Hitler, Rudolf Hess and Julius Streicher, interspersed with
footage of massed Sturmabteilung (SA) and Schutzstaffel (SS) troops and public reaction.

 Hitler commissioned the film and served as an unofficial executive producer; his name appears in the opening titles.

 The film's overriding theme is the return of Germany as a great power, with Hitler as the leader who will bring glory to the nation.

 Because the film was made after the 1934 Night of the Long Knives (on 30 June) many prominent Sturmabteilung (SA) members are absent—they were murdered in that
Party purge, organised and orchestrated by Hitler to replace the SA with the Schutzstaffel (SS) as his main paramilitary force.
</item 10>
=== END ITEM id=b3ec2f53aa7f8d3a ===

=== ITEM id=e2efb037f88fbc1f ===
Gold sentences and facts:
Sentence 0:  He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).
  Fact 0.0: The player played three seasons for the Philadelphia Eagles.
  Fact 0.1: The player's Philadelphia Eagles seasons are listed as 1940.
  Fact 0.2: The player played for the Pittsburgh Steelers.
  Fact 0.3: The player's Pittsburgh Steelers seasons are listed as 1941-1942.
Sentence 1:  He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his departure from the Eagles.
  Fact 1.0: The player was drafted by the Philadelphia Eagles.
  Fact 1.1: The player was drafted in the second round of the 2008 NFL Draft.
  Fact 1.2: The player played for the Washington Redskins for three seasons.
  Fact 1.3: The player played for the Washington Redskins after leaving the Eagles.

Retrieved items:
<item 1>
Don Looney
John Don Looney (September 2, 1916 – April 5, 2015) was a professional American football end in the National Football League.

 He was born in Sulphur Springs, Texas.

 He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).

 He was the first receiver in NFL history to have over 100 yards receiving in each of his first two games, a feat which was not equaled until the 2008 NFL
season by another Eagles wide receiver, DeSean Jackson.

 At the time of his death, Looney was the second oldest living former NFL player.

 He was the father of NFL running back Joe Don Looney, who later died in a one-person motorcycle accident after his NFL career ended.
</item 1>

<item 2>
Wilbert Montgomery
Wilbert Montgomery (born September 16, 1954) is a former American football player in the National Football League for nine years with the Philadelphia Eagles
and Detroit Lions.

 In the past, Montgomery has been the running backs and tight ends coach for St. Louis Rams (1997–2005), the running backs coach for the Detroit Lions
(2006-2007), the running backs coach for the Baltimore Ravens (2008–2013), and the running backs coach for the Cleveland Browns (2014–2015).
</item 2>

<item 3>
Dallas Texans (NFL)
The Dallas Texans played in the National Football League (NFL) for one season, 1952, with a record of 1–11.

 The team is considered one of the worst teams in NFL history, both on (lowest franchise winning percentage) and off the field.

 The team was based first in Dallas, then Hershey, Pennsylvania, and Akron, Ohio, during its only season.

 The Texans were the last NFL team to fold.

 Many players on the 1952 roster went to the new Baltimore Colts franchise in 1953.

 The American Football League (AFL) had a 1960 charter member named the Dallas Texans (who later became the Kansas City Chiefs), but the AFL Texans have no
relationship with the earlier NFL team.
</item 3>

<item 4>
1986 Tampa Bay Buccaneers season
The 1986 Tampa Bay Buccaneers season began with the team trying to improve on an 2–14 season.

 This was one of the worst seasons in franchise history.

 There is some sentiment that the 1986 team was even worse than the winless team of 1976, and the 473 points conceded was not beaten by any NFL team until the
2001 Indianapolis Colts gave up 486.

 The Buccaneers selected Bo Jackson with the top pick in the draft, but were unable to convince him to join the team.

 Three weeks after the draft, Jackson signed a three-year baseball contract with the Kansas City Royals.

 Despite holding four of the first forty selections in the draft, and the presence of a great influx of fresh talent from defunct USFL teams, the Buccaneers
were unable to find any impact players in either the draft or free agency.

 They entered the season with a roster nearly identical to the previous season's 2–14 team.
</item 4>

<item 5>
Jim O'Brien (American football)
Jim O'Brien (born February 2, 1947) is a former American football placekicker in the National Football League.

 He played for the Baltimore Colts from 1970 to 1972 and the Detroit Lions in 1973.

 He also played wide receiver, catching the bulk of his career passes during the 1972 season while still performing his kicking duties.

 His short career was less than stellar, posting a 55.6 percentage making 60 of 108 field goal attempts.

 His shining moment came in the closing moments of Super Bowl V in January 1971, where he kicked a 32-yard field goal with only five seconds remaining in the
game to break a tie and give the Colts the victory over the Dallas Cowboys 16–13.

 Before kicking the field goal, teammates saw that O'Brien was so nervous, he tried to take some of the artificial turf off the field to figure out the wind,
thinking the field was regular grass.

 Because of his singular moment kicking the Super Bowl-winning field goal, NFL Films named him the #9 "One-Hit Wonder" of all time.
</item 5>

<item 6>
Brooklyn Dodgers (NFL)
Later co-owner and partner Dan Topping (1912–1974), pulled the Tigers team out of the old NFL in 1946 and placed it in the newly established rival professional
league – the All-America Football Conference, which shortly lasted until 1949 until several stronger teams from the AAFC merged with and entered a reorganized
NFL in 1950.

 It lasted until 1970 with the NFL-AFL (third) merger following the establishment of the first "Super Bowl" inter-league national championship game three years
before with the old NFL champions playing the victors of the latest rival fourth incarnation of the American Football League IV, formed in 1960 (now the
American Football Conference (AFC).
</item 6>

<item 7>
Brooklyn Dodgers (NFL)
The Brooklyn Dodgers were an American football team that played in the National Football League from 1930 to 1943, and in 1944 as the Brooklyn Tigers.

 The team played its home games at Ebbets Field of the baseball National League's team, the Brooklyn Dodgers.

 In 1945, because of financial difficulties and the increasing scarcity of major league-level players because of the war-time defense requirements at the height
of World War II, the team was merged with the Boston Yanks and were known as the Yanks for that season.

 This old NFL franchise was not related to the earlier (second incarnation) American Football League II with a franchise that played as the Brooklyn Tigers for
the first half of the 1936 season before moving to Rochester, New York and playing as the Rochester Tigers.

 Another NFL team that played in the Brooklyn borough was the Brooklyn Lions (which became the Brooklyn Horsemen after merging with a team from an earlier first
incarnation AFL of the same name) in 1926.
</item 7>

<item 8>
Graph query row:
person1: Don Looney
relationship1: MEMBER_OF
membership_status1: None
person2: DeSean Jackson
relationship2: MEMBER_OF
membership_status2: None
team: Philadelphia Eagles
</item 8>

<item 9>
Verlon Biggs
Verlon Marion Biggs (March 16, 1943 – June 7, 1994) was an American football defensive end in the American Football League and National Football League.

 He played for the New York Jets (AFL) in Super Bowl III, but felt he didn't receive enough credit for the Jets' playoff win against the Oakland Raiders in the
AFL Championship Game that launched them into the Super Bowl.

 He played well but sulked until 1970, demanded more money, and wound up signing with Vince Lombardi's Washington Redskins of the NFL.

 Always a dominating defensive end, Verlon solidified George Allen's defense (Allen replaced Lombardi upon his death in July 1970) and led the Redskins into
Super Bowl VII.

 His nickname with the Redskins was "dirty Biggs" because of his extremely physical style of play.
</item 9>

<item 10>
DeSean Jackson
DeSean William Jackson (born December 1, 1986) is an American football wide receiver for the Tampa Bay Buccaneers of the National Football League (NFL).

 He played college football for the University of California, Berkeley, where he was recognized as a consensus All-American.

 He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his
departure from the Eagles.

 Jackson has been selected to the Pro Bowl three times, and was the first player selected to the Pro Bowl at two different positions in the same year when he
was named to the 2010 Pro Bowl as a wide receiver and return specialist.
</item 10>
=== END ITEM id=e2efb037f88fbc1f ===

=== ITEM id=0397bfc75a16e6a6 ===
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
Mosè in Egitto
Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a
1760 play by Francesco Ringhieri, "L'Osiride".

 It premièred on 5 March 1818 at the recently reconstructed Teatro San Carlo in Naples, Italy.
</item 2>

<item 3>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 3>

<item 4>
Lucrezia Borgia (1947 film)
Lucrezia Borgia (Spanish:Lucrecia Borgia) is a 1947 Argentine historical film directed by Luis Bayón Herrera and starring Olinda Bozán, Héctor Quintanilla and
Gogó Andreu.

 The film portrays the life of Lucrezia Borgia (1480-1519).
</item 4>

<item 5>
Lucrezia Borgia (1940 film)
Lucrezia Borgia is a 1940 Italian historical film directed by Hans Hinrich and starring Isa Pola, Friedrich Benfer and Carlo Ninchi.

 The film portrays the life of Lucrezia Borgia (1480-1519), one of a number of Italian films of the era set during the Renaissance.

 It was made at the Scalera Studios in Rome.
</item 5>

<item 6>
Lucrezia Borgia (play)
Lucrezia Borgia (French: "Lucrèce Borgia" ) is an 1833 play by the French writer Victor Hugo.

 It is a historical work portraying the Renaissance-era Italian aristocrat Lucrezia Borgia.

 The play (along with "Angelo, Tyrant of Padua") is believed to have been a major influence on Oscar Wilde's "The Duchess of Padua" (1891).
</item 6>

<item 7>
Lucrezia Borgia (1922 film)
Lucrezia Borgia is a 1922 German silent historical film directed by Richard Oswald and starring Conrad Veidt, Liane Haid and Albert Bassermann.

 It was based on a novel by Harry Sheff, and portrayed the life of the Renaissance Italian aristocrat Lucrezia Borgia (1480-1519).

 Botho Hoefer and Robert Neppach worked as the film's art directors, designing the period sets needed.
</item 7>

<item 8>
Lucrezia Borgia (1912 film)
Lucrezia Borgia is a 1912 Italian silent historical film directed by Gerolamo Lo Savio and starring Francesca Bertini in the title role of Lucrezia Borgia.
</item 8>

<item 9>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
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
=== END ITEM id=0397bfc75a16e6a6 ===

=== ITEM id=2a6c426f9a8b1993 ===
Gold sentences and facts:
Sentence 0:  She was also a member of a secret society composed exclusively of Cheyenne women.
  Fact 0.0: The woman in question was also a member of a secret society.
  Fact 0.1: The secret society was composed exclusively of Cheyenne women.
Sentence 1: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.
  Fact 1.0: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance.
  Fact 1.1: The other central institution of traditional Cheyenne Indian tribal governance is the Council of Forty-four.

Retrieved items:
<item 1>
Lean Bear
Lean Bear (Cheyenne name Awoninahku, 1813–1864), alternatively translated as Starving Bear, was a Cheyenne peace chief.

 He was a member of the Council of Forty-four, a tribal governance devoted to maintaining peace with encroaching United States settlers.

 Lean Bear's most notable peace deals include the Treaty of Fort Wise and a meeting with US President Abraham Lincoln.

 His work towards peace between his people and the American settlers in the Southern Plains were cut short when he was murdered by the 1st Colorado Cavalry
Regiment and violent retaliations ensued.
</item 1>

<item 2>
Council of Forty-four
The Council of Forty-four was one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the military societies such
as the Dog Soldiers.

 The influence of the Council of Forty-four waned in the face of internal conflict among the Cheyenne about Cheyenne policy toward encroaching white settlers on
the Great Plains, and was dealt a severe blow by the Sand Creek Massacre.
</item 2>

<item 3>
Ehyophsta
Ehyophsta (Cheyenne for "Yellow-Haired Woman") was a Cheyenne woman.

 She was the daughter of Stands-in-the-Timber who died in 1849, and she was the niece of Bad Faced Bull.

 She fought in the Battle of Beecher Island in 1868, and also fought the Shoshone that same year, where she counted coup against one enemy and killed another.

 She fought the Shoshone again in 1869.

 She was also a member of a secret society composed exclusively of Cheyenne women.

 She died in 1915.
</item 3>

<item 4>
Northern Cheyenne Indian Reservation
The Northern Cheyenne Indian Reservation (Tsėhéstáno in Cheyenne, formerly named the Tongue River Indian Reservation) is home of the federally recognized
Northern Cheyenne Tribe.

 Located in southeastern Montana, the reservation is approximately 444,000 acres in size and home to approximately 5,000 Cheyenne people.

 The tribal and government headquarters are in Lame Deer, which is also the home of the annual Northern Cheyenne Pow wow.

 The reservation is bounded on the east by the Tongue River and on the west by the Crow Reservation.

 There are small parcels of non-contiguous off-reservation trust lands in Meade County, South Dakota, northeast of the city of Sturgis.

 Its timbered ridges that extend into northwestern South Dakota are part of Custer National Forest and it is approximately 40 mi east of the site of the 1876
Battle of the Greasy Grass (known to most Americans as the Battle of the Little Bighorn).
</item 4>

<item 5>
Eugene Little Coyote
Eugene Little Coyote was the president of the Northern Cheyenne Indian Reservation from 2004 to 2007.

 He was elected in November 2004, defeating the incumbent president, Geri Small.

 However, after a conflict with reservation vice president Rick Wolfname that began in July 2007 escalated, the Northern Cheyenne tribal council declared that
it would oust Little Coyote as the tribe's president.

 Little Coyote was forcibly removed from office and arrested for trespassing in the tribal office building on December 28, 2007.

 It is worth noting that the move by the tribal council to remove Little Coyote from office was deemed unconstitutional and void by the Northern Cheyenne
Constitutional Court.

 The Constitutional Court was however overruled by Bureau of Indian Affairs' regional director Ed Parisian, and later the Assistant Secretary of Indian Affairs
Carl J. Artman.

 Their involvement in this matter set a historical precedence of the BIA directly meddling with tribal governments.
</item 5>

<item 6>
Chief Dull Knife College
Chief Dull Knife College (originally Dull Knife Memorial College) is a small, open-admission, Native American tribal community college and land grant
institution.

 Located on the Northern Cheyenne Indian Reservation, in the city of Lame Deer, it has a current enrollment of about 141 students.

 On average, more than half of its graduates move on to four-year colleges.
</item 6>

<item 7>
Cheyenne
The Cheyenne ( ) are one of the Indigenous peoples of the Great Plains and their language is of the Algonquian language family.

 The Cheyenne comprise two Native American tribes, the Só'taeo'o or Só'taétaneo'o (more commonly spelled as Suhtai or Sutaio) and the Tsétsêhéstâhese (also
spelled Tsitsistas).

 These tribes merged in the early 19th century.

 Today, the Cheyenne people are split into two federally recognized Nations: the Southern Cheyenne, who are enrolled in the Cheyenne and Arapaho Tribes in
Oklahoma, and the Northern Cheyenne, who are enrolled in the Northern Cheyenne Tribe of the Northern Cheyenne Indian Reservation in Montana.
</item 7>

<item 8>
Lame Deer, Montana
Lame Deer (Meaveʼhoʼeno in Cheyenne) is a census-designated place (CDP) in Rosebud County, Montana.

 The community is named after Miniconjou Lakota chief Lame Deer, who was killed by the U.S. Army in 1877 under a flag of truce south of the town.

 It is the tribal and government agency headquarters of the Northern Cheyenne Indian Reservation.

 This is the location of the Chief Dull Knife College and the annual Northern Cheyenne Powwow.
</item 8>

<item 9>
Cheyenne military societies
Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.

 While council chiefs are responsible for overall governance of individual bands and the tribe as a whole, the headmen of military societies are in charge of
maintaining discipline within the tribe, overseeing tribal hunts and ceremonies, and providing military leadership.

 Historically, council chiefs selected which of the six military societies would assume these duties; after a period of time on-duty, the chiefs would select a
different society to take up the duties.
</item 9>

<item 10>
Concho Indian Boarding School
Concho Indian Boarding School (also known as the Cheyenne-Arapaho Boarding School at Concho or Concho Indian School and home to the Concho Demonstration School)
was a boarding school for members of the Cheyenne and Arapaho Tribes and later opened to other Native American students.

 It existed from 1909 to 1983.

 It was located in central Oklahoma, approximately 1 mile south of Concho, Oklahoma and 4 miles north of El Reno, Oklahoma.

 The name of the town and school is the Spanish word for "shell" and was named for the Indian agent, Charles E. Shell.
</item 10>
=== END ITEM id=2a6c426f9a8b1993 ===

=== ITEM id=0f71528b3d4556ee ===
Gold sentences and facts:
Sentence 0: Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11, 1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical physics.
  Fact 0.0: Nikolay Mitrofanovich Krylov was born on 29 November [O.S. 17 November] 1879.
  Fact 0.1: Nikolay Mitrofanovich Krylov died on May 11, 1955.
  Fact 0.2: Nikolay Mitrofanovich Krylov's Russian name is Николай Митрофанович Крылов.
  Fact 0.3: Nikolay Mitrofanovich Krylov's Ukrainian name is Микола Митрофанович Крилов.
  Fact 0.4: Nikolay Mitrofanovich Krylov was a Russian mathematician.
  Fact 0.5: Nikolay Mitrofanovich Krylov was a Soviet mathematician.
  Fact 0.6: Nikolay Mitrofanovich Krylov is known for works on interpolation.
  Fact 0.7: Nikolay Mitrofanovich Krylov is known for works on non-linear mechanics.
  Fact 0.8: Nikolay Mitrofanovich Krylov is known for works on numerical methods for solving equations of mathematical physics.
Sentence 1: Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.
  Fact 1.0: Anatoly Timofeevich Fomenko's Russian name is Анатолий Тимофеевич Фоменко.
  Fact 1.1: Anatoly Timofeevich Fomenko was born on 13 March 1945 in Stalino, USSR.
  Fact 1.2: Anatoly Timofeevich Fomenko is a Soviet mathematician.
  Fact 1.3: Anatoly Timofeevich Fomenko is a Russian mathematician.
  Fact 1.4: Anatoly Timofeevich Fomenko is a professor at Moscow State University.
  Fact 1.5: Anatoly Timofeevich Fomenko is well known as a topologist.
  Fact 1.6: Anatoly Timofeevich Fomenko is a member of the Russian Academy of Sciences.

Retrieved items:
<item 1>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 1>

<item 2>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 2>

<item 3>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 3>

<item 4>
Nicolai V. Krylov
Bellman equations), which were examined with stochastic methods.

 This led to the Evans-Krylov theory, for which he received with Lawrence C. Evans in 2004 the Leroy P. Steele Prize of the American Mathematical Society (for
work done simultaneously and independently by both Krylov and Evans).

 They proved the second order differentiability (Hölder continuity of the second derivative) of the solutions of convex, completely nonlinear, second order
elliptical partial differential equations and thus the existence of "classical solutions" (Theorem of Evans-Krylov).

 He was in 1978 at Helsinki and in 1986 at Berkeley an Invited Speaker for the ICM.

 He received the Humboldt Research Award in 2001.

 In 1993 he was elected a member of the American Academy of Arts and Sciences (1993).

 He should not be confused with the mathematician Nikolay M. Krylov.
</item 4>

<item 5>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 5>

<item 6>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 6>

<item 7>
Graph query row:
person1: Nikolay Mitrofanovich Krylov
occupation1: mathematician
connection1: None
sharedConnection: None
connection2: None
person2: Anatoly Timofeevich Fomenko
occupation2: ['mathematician', 'professor', 'topologist']
</item 7>

<item 8>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 8>

<item 9>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 9>

<item 10>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 10>
=== END ITEM id=0f71528b3d4556ee ===

=== ITEM id=9abd71b3ac54ba55 ===
Gold sentences and facts:
Sentence 0:  Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
  Fact 0.0: The unnamed person in this passage has other voice-over roles
  Fact 0.1: The unnamed person voiced Dr. Drakken in Kim Possible
  Fact 0.2: The unnamed person voiced Motor Ed in Kim Possible
  Fact 0.3: The unnamed person voiced Brother Blood in Teen Titans
  Fact 0.4: The unnamed person voiced Aquaman in an unnamed work
  Fact 0.5: The unnamed person voiced Rico in The Penguins of Madagascar
  Fact 0.6: The unnamed person voiced Ogre in an unnamed work
  Fact 0.7: The unnamed person voiced Niblet in Pound Puppies
  Fact 0.8: The unnamed person voiced the Scotsman in Samurai Jack
  Fact 0.9: The unnamed person voiced Shnitzel in Chowder
Sentence 1: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 1.0: Chowder is an American animated television series
  Fact 1.1: Chowder was created by C. H. Greenblatt
  Fact 1.2: Chowder was created for Cartoon Network

Retrieved items:
<item 1>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 1>

<item 2>
Sherry Alberoni
Sharyn Eileen "Sherry" Alberoni (born December 4, 1946) is an American actress and voice artist.

 Alberoni got her start as a Mouseketeer on the weekday ABC television program "The Mickey Mouse Club".

 As an adult, she became a voice artist for Hanna-Barbera Productions.

 Besides providing voices for numerous incidental characters in series such as "Jeannie", Alberoni is best known as the voice of nasty rich-girl Alexandra Cabot
from "Josie and the Pussycats" , "superhero-in-training" Wendy from the first season of "Super Friends", the heroic robot, Bo in "Mighty Orbots" and
Glumdalclitch in "The Three Worlds of Gulliver".

 In 1971, she starred alongside Patty Andrews in the Sherman Brothers stage musical, "Victory Canteen".
</item 2>

<item 3>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 3>

<item 4>
Adventure Time
Adventure Time is an American animated television series created by Pendleton Ward for Cartoon Network.

 It follows the adventures of a boy named Finn (voiced by Jeremy Shada) and his best friend and adoptive brother Jake (John DiMaggio)— a dog with the magical
power to change shape and size at will.

 Finn and Jake live in the post-apocalyptic Land of Ooo, where they interact with Princess Bubblegum (Hynden Walch), the Ice King (Tom Kenny), Marceline the
Vampire Queen (Olivia Olson), BMO (Niki Yang), and others.

 The series is based on a 2007 short produced for Nicktoons and Frederator Studios' animation incubator series "Random!

 Cartoons".

 After the short became a viral hit on the Internet, Cartoon Network commissioned a full-length series, which previewed on March 11, 2010, and officially
premiered on April 5, 2010.
</item 4>

<item 5>
Kenny Hotz
Kenneth Joel "Kenny" Hotz is a Canadian comedy writer, producer, director, actor and comedian.

 Hotz is a "South Park" consultant and writer, creator/star of the Comedy Central television show "Kenny vs. Spenny", creator and writer of the FX series
"Testees", and the creator/star of "Kenny Hotz's Triumph of the Will".

 He has received numerous awards for his television work— from the Academy of Canadian Cinema & Television the Gemini Awards and the Canadian Comedy Awards—and
is a multiple-time film festival award-winner.

 He is also an award-winning "Vice" contributor and Gulf war photo-journalist.
</item 5>

<item 6>
List of Invader Zim characters
"Invader Zim" is an American animated television series created by Jhonen Vasquez and originally aired on Nickelodeon.

 The recurring cast includes voice actors Richard Steven Horvitz, Rodger Bumpass, Kevin McDonald ("The Kids in the Hall") and John de Lancie ("").

 Vasquez voices parts in the show.

 The show was cancelled early in its run and some episodes were unfinished.

 The show ran for two seasons before its cancellation.
</item 6>

<item 7>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 7>

<item 8>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 8>

<item 9>
Adventure Time (pilot)
"Adventure Time" is an animated short created by Pendleton Ward, as well as the pilot to the Cartoon Network series of the same name.

 The short follows the adventures of Pen (voiced by Zack Shada), a human boy, and his best friend Jake (voiced by John DiMaggio), a dog with magical powers to
change shape and grow and shrink at will.

 In this episode, Pen and Jake have to rescue Princess Bubblegum (voiced by Paige Moss) from the antagonistic Ice King (voiced by John Kassir).
</item 9>

<item 10>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 10>
=== END ITEM id=9abd71b3ac54ba55 ===

=== ITEM id=5ffd59af55a94163 ===
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
Information (Berlin album)
Information is the first album by Berlin, released in 1980 by Vinyl Records.

 It was recorded during the period when Terri Nunn had temporarily left the band to pursue an acting career and Virginia Macolino performed the lead vocals.

 Several songs were written by previous lead singer Toni Childs who went solo prior to the album's release.
</item 1>

<item 2>
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
</item 2>

<item 3>
Inner Secrets
Inner Secrets is the ninth studio album by Santana.

 It was released in 1978 and marks the start of the phase of Santana's career where he moved away from the fusion of Latin, jazz, rock and blues that marked his
previous records and began to move towards an album-oriented rock direction.

 As such, the album's quality is widely disputed among fans. "

Stormy" and "One Chain (Don't Make No Prison)" were both hit singles.

 In The Netherlands "Well All Right" was released as a single and reached #22 in the top 40.
</item 3>

<item 4>
Seven (Poco album)
Seven is the sixth studio album (seventh overall) by American country rock band Poco.

 It is the first album they made after leader Richie Furay left the band.

 The front cover was designed by Phil Hartman.
</item 4>

<item 5>
The Secret (Vib Gyor EP)
The Secret EP is the EP from the Leeds band Vib Gyor.

 It was released on 5 March 2007, and is the follow-up to their well-received debut single, "Fallen".

 It was recorded in Bath and produced by Paul Corkett.

 It also has a hidden fifth instrumental track - which the band recently revealed was called <:.

~.

:>
</item 5>

<item 6>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 6>

<item 7>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 7>

<item 8>
- **Quiet Riot 10** (2014), the band’s twelfth studio album and first since reuniting in 2010, features Jizzy Pearl on six new studio tracks; its final four
tracks are live performances with Kevin DuBrow, who died in 2007.
- **Inner Secrets** (1978), Santana’s ninth album, shifted toward album-oriented rock. “Stormy” and “One Chain” were hits; “Well All Right” reached No.
</item 8>

<item 9>
The entries cover diverse music milestones: Marc Ribot’s 2005 Albert Ayler tribute, *Spiritual Unity*; comic duo Collins & Harlan, whose 1916 recording helped
introduce “jas,” precursor to “jazz”; and reggae singer Cedric Myton, a Royal Rasses and Congos member whose voice features on *Humanity* and *Heart of the
Congos*. They also describe early Dean Martin, Chico Hamilton, Shelly Manne, and Maynard
</item 9>

<item 10>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 10>
=== END ITEM id=5ffd59af55a94163 ===
