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

=== ITEM id=46b9753f7842196f ===
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
Hans Aschenborn
Hans Anton Aschenborn (1 February 1888 – 10 April 1931) was a renowned animal painter of African wildlife.

 He worked both in Germany and in southern Africa.

 His work is featured in the older German Thieme-Becker or Saur art encyclopedia.

 A Master of Arts thesis by Karin Skawran concerning the graphic works of Hans Anton Aschenborn was published in the South African art and culture periodical,
"Lantern" in 1965.)

 In 1963 the University of Pretoria published a book about Aschenborn as an artist entitled, "Hans Anton Aschenborn : Mens en Kunstenaar".

 In 1970 another one followed by the Pretoria Art Museum (South Africa).

 Other publications on Aschenborn feature his etchings and linocuts.
</item 1>

<item 2>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 2>

<item 3>
Hans von Hentig
Hans von Hentig (9 June 1887 in Berlin – 6 July 1974 in Bad Tölz) was a German criminal psychologist and politician.He was the second son of lawyer Otto von
Hentig (1852–1934) .

His older brother was later diplomat Werner Otto von Hentig.

 Otto von Hentig was one of the leading lawyers in Berlin.

 Hans von Hentig was instrumental in the setting up of a short-lived Bavarian Soviet Republic in 1919.

 During the 1920s he was a prominent exponent of National Bolshevism.

 He emigrated to United States in 1935.

 Hans von Hentig worked on Yale University.
</item 3>

<item 4>
Karl Anselm, 4th Prince of Thurn and Taxis
Karl Anselm, 4th Prince of Thurn and Taxis, full German name: "Karl Anselm Fürst von Thurn und Taxis" (2 June 1733, Frankfurt am Main, Free Imperial City of
Frankfurt, Holy Roman Empire – 13 November 1805, Winzer bei Regensburg, Electorate of Bavaria, Holy Roman Empire) was the fourth Prince of Thurn and Taxis,
Postmaster General of the Imperial Reichspost, and Head of the Princely House of Thurn and Taxis from 17 March 1773 until his death on 13 November 1805.

 Karl Anselm served as "Prinzipalkommissar" at the Perpetual Imperial Diet in Regensburg for Joseph II, Holy Roman Emperor and Francis II, Holy Roman Emperor
from 1773 to 1797.
</item 4>

<item 5>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 5>

<item 6>
Richard Genée
Franz Friedrich Richard Genée (February 7, 1823 – June 15, 1895) was a Prussian born Austrian librettist, playwright, and composer.

 He is most famous for the libretto of "Die Fledermaus", Johann Strauss II's most famous operetta.

 He co-wrote the libretto without having met top-billed librettist Karl Haffner, who constructed the new story based on a play by Henri Meilhac and Ludovic
Halévy, which was considered too shocking to perform outside Paris.

 Genée, however, wrote the operetta's actual text and drew nothing from Haffner beyond the names of the characters.
</item 6>

<item 7>
Karl Wilhelm Friedrich Schlegel
Karl Wilhelm Friedrich (after 1814: von) Schlegel (10 March 1772 – 12 January 1829), usually cited as Friedrich Schlegel, was a German poet, literary critic,
philosopher, philologist and Indologist.

 With his older brother, August Wilhelm Schlegel, he was one of the main figures of the Jena romantics.

 He was a zealous promoter of the Romantic movement and inspired Samuel Taylor Coleridge, Adam Mickiewicz and Kazimierz Brodziński.

 Schlegel was a pioneer in Indo-European studies, comparative linguistics, morphological typology, and was the first to notice what became known as Grimm's law.
</item 7>

<item 8>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 8>

<item 9>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 9>

<item 10>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 10>
=== END ITEM id=46b9753f7842196f ===

=== ITEM id=0efe5dc492038d5b ===
Gold sentences and facts:
Sentence 0: Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.
  Fact 0.0: Hayley Nichole Williams was born on December 27, 1988.
  Fact 0.1: Hayley Nichole Williams is an American singer.
  Fact 0.2: Hayley Nichole Williams is an American songwriter.
  Fact 0.3: Hayley Nichole Williams is an American musician.
Sentence 1:  The band consists of Hayley Williams, Zac Farro and Taylor York.
  Fact 1.0: The band consists of Hayley Williams.
  Fact 1.1: The band consists of Zac Farro.
  Fact 1.2: The band consists of Taylor York.
Sentence 2: Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American heavy metal bands Lord Tracy and Pantera.
  Fact 2.0: Terrence Lee Glaze was born on November 29, 1964, in Columbus.
  Fact 2.1: Terrence Lee Glaze is known as Terry Lee.
  Fact 2.2: Terry Lee is a Texas-raised singer.
  Fact 2.3: Terry Lee is a Texas-raised guitarist.
  Fact 2.4: Terry Lee is most known for his work in the American heavy metal band Lord Tracy.
  Fact 2.5: Terry Lee is most known for his work in the American heavy metal band Pantera.

Retrieved items:
<item 1>
Kate McTell
Kate McTell (born Ruthy Kate Williams; August 22, 1911 – October 3, 1991) was an American blues musician and nurse from Jefferson County, Georgia.

 She is known primarily as the former wife of the blues musician Blind Willie McTell, whom she accompanied vocally on several recordings.

 She may have recorded as Ruby Glaze, but there is some uncertainty about whether she and Glaze were the same person, despite the fact that she claimed to be
Glaze.
</item 1>

<item 2>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 2>

<item 3>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 3>

<item 4>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 4>

<item 5>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 5>

<item 6>
Hayley Williams (ice hockey)
Hayley Williams (born June 3, 1990) is an American professional ice hockey player for the Brampton Thunder of the Canadian Women's Hockey League (CWHL).

 Williams has previously played for the Buffalo Beauts in the National Women's Hockey League (NWHL).
</item 6>

<item 7>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
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
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 9>

<item 10>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 10>
=== END ITEM id=0efe5dc492038d5b ===

=== ITEM id=1491e80d3ed9487e ===
Gold sentences and facts:
Sentence 0: Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.
  Fact 0.0: Buried Country is a highly regarded documentary film
  Fact 0.1: Buried Country is a highly regarded book
  Fact 0.2: Buried Country is a highly regarded soundtrack album
  Fact 0.3: Buried Country is now also a stageshow
Sentence 1: South of the Border is a 2009 American documentary film directed by Oliver Stone.
  Fact 1.0: South of the Border is a documentary film
  Fact 1.1: South of the Border is American
  Fact 1.2: South of the Border is from 2009
  Fact 1.3: South of the Border was directed by Oliver Stone

Retrieved items:
<item 1>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 1>

<item 2>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 2>

<item 3>
Station to Station (film)
Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.

 It was distributed internationally in 2015 following its release at the Sundance Film Festival.
</item 3>

<item 4>
Plain Clothes Theatre Productions
Plain Clothes Theatre Productions is a Bristol-based theatre company producing contemporary plays from around the globe.

 Formed in 2003 by artistic director Sam Berger, the company has produced work in London, Bristol, Cheltenham, Toronto and Vancouver.

 Their work has included plays by Joe Orton, Laura Wade and David Mamet, and their style is based around the teachings of American acting coach Sanford Meisner.
</item 4>

<item 5>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 5>

<item 6>
The Colleen Bawn
The Colleen Bawn, or The Brides of Garryowen is a melodramatic play written by Irish playwright Dion Boucicault.

 It was first performed at Miss Laura Keene's Theatre, New York, on 27 March 1860 with Laura Keene playing Anne Chute and Boucicault playing Myles na Coppaleen.

 It was most recently performed in Dublin at the Project Arts Centre in July and August 2010.

 Several film versions have also been made.
</item 6>

<item 7>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 7>

<item 8>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 8>

<item 9>
Buried Country
Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.

 It is a prosopography, created by noted writer Clinton Walker, that tells the story of Australian country music in the Aboriginal community by focussing on the
genre's most important stars.
</item 9>

<item 10>
Jim Sharman
James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more
than 70 productions to his credit.

 He is renowned in Australia for his work as a theatre director from the 1960s to the present, and is best known internationally as the director of the 1973
theatrical hit "The Rocky Horror Show", its film adaptation "The Rocky Horror Picture Show" (1975) and the film's follow-up "Shock Treatment" (1981).
</item 10>
=== END ITEM id=1491e80d3ed9487e ===

=== ITEM id=2d93c3ac3a113498 ===
Gold sentences and facts:
Sentence 0: Sasson Gabai (Hebrew: ששון גבאי‎ ‎ ; born 24 November 1947) is an Israeli actor.
  Fact 0.0: Sasson Gabai's Hebrew name is ששון גבאי.
  Fact 0.1: Sasson Gabai was born on 24 November 1947.
  Fact 0.2: Sasson Gabai is an Israeli actor.
Sentence 1:  Winner of an Ophir Award, Asia Pacific Screen Award (nominated), European Film Awards and Jerusalem Film Festival in 2007 as Best Actor in a highly acclaimed Israeli film "The Band's Visit".
  Fact 1.0: Sasson Gabai won an Ophir Award.
  Fact 1.1: Sasson Gabai was nominated for an Asia Pacific Screen Award.
  Fact 1.2: Sasson Gabai won a European Film Awards award.
  Fact 1.3: Sasson Gabai won a Jerusalem Film Festival award.
  Fact 1.4: These awards were won in 2007.
  Fact 1.5: The awards were for Best Actor.
  Fact 1.6: The award-winning role was in the highly acclaimed Israeli film The Band's Visit.
Sentence 2: The Band's Visit (Hebrew: ביקור התזמורת - "Bikur Ha-Tizmoret") is a 2007 Israeli film directed by Eran Kolirin.
  Fact 2.0: The Band's Visit is a 2007 Israeli film.
  Fact 2.1: The Band's Visit is written in Hebrew as ביקור התזמורת.
  Fact 2.2: The Hebrew title of The Band's Visit is Bikur Ha-Tizmoret.
  Fact 2.3: The Band's Visit was directed by Eran Kolirin.

Retrieved items:
<item 1>
Beyond the Sea (1991 film)
Beyond the Sea (translit.

 Me'ever Layam) is a 1991 Israeli drama film directed by Jacob Goldwasser.

 It stars Aryeh Moskona, Dafna Rechter, Moti Giladi.

 The film, a family drama set in the 1960s, was critically acclaimed, and won the Ophir Award for Best Film.

 The film was selected as the Israeli entry for the Best Foreign Language Film at the 64th Academy Awards, but was not accepted as a nominee.
</item 1>

<item 2>
Hunting Elephants
Hunting Elephants (Hebrew: לצוד פילים‎ ‎ , "Latzud Pilim", "lit."

 To Hunt Elephants) is a 2013 Israeli crime comedy film directed by Reshef Levi.

 It stars, among others, Sasson Gabai, Moni Moshonov and Patrick Stewart.

 It was released to theaters in Israel on July 4, 2013.
</item 2>

<item 3>
Joseph Cedar
Yossef (Joseph) Cedar (Hebrew: יוסף סידר; born August 31, 1968) is an Israeli film director and screenwriter.

 He has won a Silver Bear and an Ophir Award for Best Director, and an Ophir Award for writing a Best Screenplay.

 He also won the best screenplay award at the 2011 Cannes Film Festival for his film "Footnote" (2011).
</item 3>

<item 4>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 4>

<item 5>
Yana's Friends
Yana's Friends (Hebrew: החברים של יאנה, "HaHaverim shel Yana"‎ ‎ ) is a 1999 Israeli film directed by Arik Kaplun.

 script editor: Savi Gabizon.

 Critically acclaimed, it won 10 Israeli Academy Awards including the Ophir Award for Best Picture.

 It also won the Crystal Globe at the Karlovy Vary International Film Festival in 1999.

 The film has a very rare 100% rating on the film website Rotten Tomatoes based on 30 reviews.
</item 5>

<item 6>
The Farewell Party
The Farewell Party (Hebrew: מיתה טובה‎ ‎ Mita Tova) is a 2014 Israeli drama film about the use of a euthanasia device, directed by Tal Granit and Sharon Maymon.

 It was nominated for the Ophir Award for Best Film.

 The film was screened in the Venice Days section of the 71st Venice International Film Festival and has been selected to be screened in the Contemporary World
Cinema section at the 2014 Toronto International Film Festival.

 The film was also screened in the 44th edition of the International Film Festival Rotterdam, reaching the second place in the IFFR audience award.
</item 6>

<item 7>
Sh'Chur
Sh'Chur is a 1994 Israeli drama film starring Gila Almagor, Ronit Elkabetz and Hanna Azoulay-Hasfari.

 It was written by Hanna Azoulay Hasfari and directed by her partner Shmuel Hasfari.

 Sh'Chur received critical acclaim and was the 1994 official Israeli submission for the Academy Award for Best Foreign Language Film.

 It was also awarded the Ophir Award for best film by the Israeli Film and Television Academy.

 After its release, the film garnered various discussions in the Israeli press over its representation of the Moroccan community in Israel.
</item 7>

<item 8>
The Band's Visit
The Band's Visit (Hebrew: ביקור התזמורת - "Bikur Ha-Tizmoret") is a 2007 Israeli film directed by Eran Kolirin.
</item 8>

<item 9>
Sasson Gabai
Sasson Gabai (Hebrew: ששון גבאי‎ ‎ ; born 24 November 1947) is an Israeli actor.

 Winner of an Ophir Award, Asia Pacific Screen Award (nominated), European Film Awards and Jerusalem Film Festival in 2007 as Best Actor in a highly acclaimed
Israeli film "The Band's Visit".
</item 9>

<item 10>
Foxtrot (2017 film)
Foxtrot (Hebrew: פוֹקְסטְרוֹט‎ ‎ ) is a 2017 Israeli drama film directed by Samuel Maoz.

 It was screened in the competition section of the 74th Venice International Film Festival where it won the Grand Jury Prize Silver Lion.

 It was also screened in the Special Presentations section at the 2017 Toronto International Film Festival.

 It won the Ophir Award for Best Film, therefore becoming the Israeli entry for the Best Foreign Language Film at the 90th Academy Awards.
</item 10>
=== END ITEM id=2d93c3ac3a113498 ===

=== ITEM id=1dd0995ccad4991b ===
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
1st Virginia State Regiment
The 1st Virginia State Regiment was a regiment of Virginia Militia that fought during the American Revolutionary War.

 The regiment was authorized by the General Assembly of the Commonwealth of Virginia in December 1776 as a force of regular troops for the Commonwealth's
defense.

 In 1777, Virginia had difficulty meeting its quota for the regular line of the Continental Army.

 As a result, in July 1777 under the command of Colonel George Gibson, the regiment began a march North to temporarily join the Continental Army in the
Philadelphia Campaign.

 In January 1778, the Virginia General Assembly passed an act directing that the 1st Virginia State Regiment "now in Continental service, be continued in said
service instead of the Ninth Virginia Regiment, made prisoners by the enemy in the Battle of Germantown."

 The regiment camped at Valley Forge in the winter of 1777-78 and at Middlebrook in the winter of 1778-79 and participated in the Battle of Monmouth.

 The regiment remained in the service of the Continental Army until late 1779 when redeployed Virginia.
</item 1>

<item 2>
William King (governor)
William King (February 9, 1768June 17, 1852) was an American merchant, shipbuilder, army officer, and statesman from Bath, Maine.

 A proponent of statehood for Maine, he became its first governor when it separated from Massachusetts in 1820.
</item 2>

<item 3>
Maine gubernatorial election, 1820
The 1820 Maine gubernatorial election took place on April 3, 1820.

 It was the first election for Governor of Maine, taking place after Maine separated from Massachusetts and was recognized as a state on March 15, 1820.

 Maine's separation from Massachusetts came as a result of The Missouri Compromise.

 This election saw the virtually unanimous election of William King, the man most chiefly responsible for the push for Maine statehood.

 He had no opponents.
</item 3>

<item 4>
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
Parallel 36°30′ north
The parallel 36°30′ north is a circle of latitude that is 36 and one-half degrees north of the equator of the Earth.

 This parallel of latitude is particularly significant in the history of the United States as the line of the Missouri Compromise, which was used to divide the
prospective slave and free states west of the Mississippi River, with the exception of Missouri, which is mostly north of this parallel.
</item 6>

<item 7>
Anti-Nebraska movement
The Anti-Nebraska movement was a political alignment in the United States formed in opposition to the Kansas-Nebraska Act of 1854 and to its repeal of the
Missouri Compromise provision forbidding slavery in U.S. territories north of latitude 36° 30' N.

 (At the time, the name "Nebraska" could loosely refer to areas west of the Missouri River.)
</item 7>

<item 8>
Dred Scott
Dred Scott (c. 1799 – September 17, 1858) was an enslaved African American man in the United States who unsuccessfully sued for his freedom and that of his wife
and their two daughters in the "Dred Scott v. Sandford" case of 1857, popularly known as the "Dred Scott Decision".

 Scott claimed that he and his wife should be granted their freedom because they had lived in Illinois and the Wisconsin Territory for four years, where slavery
was illegal.

 The United States Supreme Court decided 7–2 against Scott, finding that neither he nor any other person of African ancestry could claim citizenship in the
United States, and therefore Scott could not bring suit in federal court under diversity of citizenship rules.

 Moreover, Scott's temporary residence outside Missouri did not bring about his emancipation under the Missouri Compromise, which the court ruled
unconstitutional as it would "improperly deprive Scott's owner of his legal property".
</item 8>

<item 9>
Pottawatomie massacre
The Pottawatomie massacre occurred during the night of May 24 and the morning of May 25, 1856.

 In reaction to the sacking of Lawrence, Kansas by pro-slavery forces, John Brown and a band of abolitionist settlers—some of them members of the Pottawatomie
Rifles—killed five settlers north of Pottawatomie Creek in Franklin County, Kansas.

 This was one of the many bloody episodes in Kansas preceding the American Civil War, which came to be known collectively as Bleeding Kansas.

 Bleeding Kansas was largely brought about by the Missouri Compromise and the Kansas–Nebraska Act.
</item 9>

<item 10>
Henry Smith Lane
Henry Smith Lane (February 24, 1811 – June 19, 1881) was a United States Representative, Senator, and the 13th Governor of Indiana; he was by design the
shortest-serving Governor of Indiana, having made plans to resign the office should his party take control of the Indiana General Assembly and elect him to the
United States Senate.

 He held that office for only two days, and was known for his opposition to slavery.

 A Whig until the party collapsed, he supported compromise with the south.

 He became an early leader in the Republican Party starting in 1856 serving as the president of the first party convention, delivering its keynote address, and
was influential in the nomination of Abraham Lincoln.

 With the repeal of the Missouri Compromise, he became a full-fledged
</item 10>
=== END ITEM id=1dd0995ccad4991b ===

=== ITEM id=ef83bb939035ca2e ===
Gold sentences and facts:
Sentence 0: Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on the German island of Greifswalder Oie in December 1937.
  Fact 0.0: Operation Lighthouse was the name given to a failed experimental launch.
  Fact 0.1: The Operation Lighthouse launch involved four Aggregate 3 liquid-fuel rockets.
  Fact 0.2: Wernher von Braun conducted the Operation Lighthouse launch.
  Fact 0.3: Walter Dornberger conducted the Operation Lighthouse launch.
  Fact 0.4: The Operation Lighthouse launch took place on the German island of Greifswalder Oie.
  Fact 0.5: The Operation Lighthouse launch took place in December 1937.
Sentence 1: Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War II.
  Fact 1.0: Major-General Dr. Walter Robert Dornberger was born on 6 September 1895.
  Fact 1.1: Walter Robert Dornberger died on 27 June 1980.
  Fact 1.2: Walter Robert Dornberger was a German Army artillery officer.
  Fact 1.3: Walter Robert Dornberger's career spanned World War I.
  Fact 1.4: Walter Robert Dornberger's career spanned World War II.

Retrieved items:
<item 1>
Magnus von Braun
Magnus "Mac" Freiherr von Braun (10 May 1919 – 21 June 2003) was a German chemical engineer, Luftwaffe aviator, and rocket scientist at Peenemünde, the
Mittelwerk, and after emigrating to the United States via Operation Paperclip, at Fort Bliss.

 He was the brother of Sigismund and Wernher von Braun.
</item 1>

<item 2>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 2>

<item 3>
Mount Von Braun
Mount Von Braun ( ) is a mountain (3,275 m) located 4 nautical miles (7 km) south of Mount Sabine in the Admiralty Mountains.

 Mapped by United States Geological Survey (USGS) from surveys and U.S. Navy air photos, 1960–63.

 Named by Advisory Committee on Antarctic Names (US-ACAN) for Wernher von Braun of the National Aeronautics and Space Administration, a visitor at McMurdo
Station, 1966–67.
</item 3>

<item 4>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 4>

<item 5>
Man Will Conquer Space Soon!
Man Will Conquer Space Soon!

 was the title of a famous series of 1950s magazine articles in "Collier's" detailing Wernher von Braun's plans for manned spaceflight.

 Edited by Cornelius Ryan, the individual articles were authored by such space notables of the time as Willy Ley, Fred Lawrence Whipple, Dr. Joseph Kaplan, Dr.
Heinz Haber, and von Braun.

 The articles were illustrated with paintings and drawings by Chesley Bonestell, Fred Freeman, and Rolf Klep, some of the finest magazine illustrators of the
time.
</item 5>

<item 6>
Von Braun (crater)
von Braun, named after the rocket pioneer Wernher von Braun, is a lunar crater located near the northwestern limb of the Moon.

 It lies along the western edge of the Oceanus Procellarum, to the northeast of the crater Lavoisier.

 The northeastern rim of this crater is on the edge of the Sinus Roris, a bay feature in the northwestern part of the Oceanus Procellarum.

 Due to its proximity to the limb, this crater appears significantly foreshortened when viewed from the Earth.
</item 6>

<item 7>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 7>

<item 8>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 8>

<item 9>
Von Braun Ferry Rocket
Von Braun Ferry Rocket was a concept design for a shuttle spacecraft that was developed by Wernher von Braun in a seminal series of early-1950s Collier's
magazine articles, Man Will Conquer Space Soon!

 by Wernher von Braun "et al." The Ferry Rocket concept has evolved over time.
</item 9>

<item 10>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 10>
=== END ITEM id=ef83bb939035ca2e ===

=== ITEM id=ed4ad9ae165f2e94 ===
Gold sentences and facts:
Sentence 0: Black (Bengali: ব্ল্যাক ) is a Bangladeshi rock band, formed in Dhaka, Bangladesh.
  Fact 0.0: Black is a Bangladeshi rock band.
  Fact 0.1: Black was formed in Dhaka, Bangladesh.
  Fact 0.2: The Bengali name of Black is ব্ল্যাক.
Sentence 1: New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.
  Fact 1.0: New Found Glory is an American rock band.
  Fact 1.1: New Found Glory was formerly known as A New Found Glory.
  Fact 1.2: New Found Glory is from Coral Springs, Florida.
  Fact 1.3: New Found Glory was formed in 1997.

Retrieved items:
<item 1>
From the Screen to Your Stereo
From the Screen to Your Stereo is an EP by A New Found Glory (now New Found Glory) released on 28 March 2000 by Drive-Thru Records.

 All the tracks are covers of songs from motion picture soundtracks.

 The album was pressed on 10" vinyl and released in three different colors: light marble blue, marble red, and white.

 The vinyl pressing featured a bonus track.

 In 2002, the album was released on a non-US split with the Japanese band Nicotine entitled Movie Addiction, where on this split, Nicotine also covered songs
from movies.

 In 2007 the band released a follow up, this time a full-length album titled "From the Screen to Your Stereo Part II".
</item 1>

<item 2>
A Very New Found Glory Christmas
A Very New Found Glory Christmas is the third EP by American rock band New Found Glory.

 Self-produced by the band, it is a special edition Christmas-themed acoustic recording featuring two original songs and three covers.

 The EP was limited to 2,000 pressings and was released via cassette tape on December 1, 2012.

 1,000 white copies were sold on tour, and 1,000 red copies were sold in their online store.

 The band released "Nothing for Christmas" via Rdio and Spotify on December 20, after debuting songs off the EP during the tenth anniversary tour of their
"Sticks and Stones" album.

 The cover art is a parody of the 1990 blockbuster movie "Home Alone".
</item 2>

<item 3>
It's All About the Girls
It's All About the Girls is the debut EP by A New Found Glory (now New Found Glory) released on December 20, 1997 by Fiddler Records.

 It is the only release with their original drummer Joe Marino.

 A re-release, featuring revised cover artwork, was issued by Fiddler in June, 2000.
</item 3>

<item 4>
Nothing Gold Can Stay (album)
Nothing Gold Can Stay is the debut studio album by American rock band New Found Glory, released on October 19, 1999 through independent record label Eulogy
Recordings.

 At the time, the band was then named "A New Found Glory", but later dropped the indefinite article "A" due to some fans struggling to find their records in
stores.

 The original pressings of the album contained samplings from several films including "The Outsiders" (1983), "Weird Science" (1985), and "That Thing You Do!

" (1996), as well as Robert Frost's poem "Nothing Gold Can Stay", after which the album is named.
</item 4>

<item 5>
Chad Gilbert
Chad Everett Gilbert (born March 9, 1981) is an American musician and record producer.

 He is a founding member of the rock band New Found Glory, for whom he plays lead guitar, composes music and provides backing vocals.

 He was also the lead vocalist for New Found Glory's now-defunct side-project, International Superheroes of Hardcore.

 Additionally, Gilbert was the vocalist for the hardcore punk band Shai Hulud between 1995 and 1998, and he rejoined the band in 2012.
</item 5>

<item 6>
Glassjaw
Glassjaw is an American post-hardcore band from Long Island, New York.

 The band is fronted by vocalist Daryl Palumbo and guitarist Justin Beck.

 The band has been influential in the progression of the underground music scene in the eastern United States and United Kingdom and on the post-hardcore genre,
and are known for their intense live shows and frequent line-up changes.
</item 6>

<item 7>
Clearview 77
Clearview 77 was a punk rock quartet hailing from Brooklyn, NY.

 Born in the summer of 1996, CV77 was the true definition of the "local" band.

 The band has played countless shows within the East Coast with acts like Face to Face, New Found Glory, Finch, Saves the Day, The Amazing Crowns, US Bombs, Big
Wig, Murphy's Law, Alkaline Trio, The Queers, GOB, and many more.

 In their 8-year existence the band managed to finish at the top of the charts and competitions for CMJ, Discmakers, Garageband.com, mp3.com, Jim Beam,
Loudenergy, had a song aired on CNN and Good Morning America, and added to college radio stations across the US.

 The band also finished in the top 10 in 92.3 FM K-ROCK' best unsigned band of 2000 contest.

 Clearview 77 have also been featured on multiple compilations with acts like NOFX, New Found Glory, Auto Pilot Off, Agnostic Front, Midtown, ALL, Saves the
Day, Down By Law, Circle Jerks, and many more.
</item 7>

<item 8>
Hit or Miss (New Found Glory song)
"Hit or Miss (Waited Too Long)" is the first single by New Found Glory (formerly A New Found Glory), from their 1999 debut studio album, "Nothing Gold Can
Stay".
</item 8>

<item 9>
Black (Bangladeshi band)
Black (Bengali: ব্ল্যাক ) is a Bangladeshi rock band, formed in Dhaka, Bangladesh.
</item 9>

<item 10>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 10>
=== END ITEM id=ed4ad9ae165f2e94 ===

=== ITEM id=d84ab90b77571b61 ===
Gold sentences and facts:
Sentence 0: Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.
  Fact 0.0: Jeffrey Charles "Jeff" Ragsdale is an American author.
  Fact 0.1: Jeffrey Charles "Jeff" Ragsdale is a documentary filmmaker.
  Fact 0.2: Jeffrey Charles "Jeff" Ragsdale is an actor.
  Fact 0.3: Jeffrey Charles "Jeff" Ragsdale is a stand-up comedian.
Sentence 1: Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.
  Fact 1.0: Millard Webb was born on 6 December 1893.
  Fact 1.1: Millard Webb died on 21 April 1935.
  Fact 1.2: Millard Webb was an American screenwriter.
  Fact 1.3: Millard Webb was an American director.

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
Howard R. Garis
Howard Roger Garis ((1873--) 25, 1873 – (1962--) 6, 1962 ) was an American author, best known for a series of books, published under his own name, that featured
the character of Uncle Wiggily Longears, an engaging elderly rabbit.

 Garis and his wife were possibly the most prolific children's authors of the early 20th century.

 Many of his books were illustrated by Lansing Campbell.
</item 2>

<item 3>
Graph query row:
person: Millard Webb
work: The Happy Ending
relationship: CREATED_BY
</item 3>

<item 4>
Jeff Ragsdale
Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.

 In 2011 he posted a flyer in New York City as a "social experiment", stating his phone number and asking people to call him, describing himself as "Jeff, one
lonely guy".

 He was overwhelmed with thousands of calls after photos of the flyer were posted on the internet.

 The experience led to his 2012 book "Jeff, One Lonely Guy", and indirectly to a 2013 pilot episode for a reality television show, "Being Noticed", and a
starring role in the 2014 documentary "Hotline".
</item 4>

<item 5>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 5>

<item 6>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 6>

<item 7>
Ross Winans
Ross Winans (1796–1877) was an American inventor, mechanic, and builder of locomotives and railroad machinery.

 He is also noted for design of pioneering cigar-hulled ships.

 Winans, one of the United States' first multi-millionaires, was involved in national and state politics, a southern-sympathizer and was a vehement "states'
rights" advocate.

 His outspoken anti-federal stance as a member of the Maryland House of Delegates, the lower chamber of the General Assembly, (state legislature) led to his
temporary arrest on board a Baltimore and Ohio Railroad train returning from an early session of the legislature held in the western Maryland town of Frederick
to avoid the Union Army-occupied state capital of Annapolis in April–May, 1861, to consider the possibilities of state secession, during the early decisive
period of the American Civil War.

 Winans was related to James McNeill Whistler through marriage (Whistler's brother George married Winans' daughter Julia).
</item 7>

<item 8>
Millard Webb
Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.

 He directed 20 films between 1920 and 1933.

 His best-known film is the 1926 silent John Barrymore adventure "The Sea Beast" costarring Dolores Costello.

 Webb also directed the early sound Florenz Ziegfeld produced talkie "Glorifying the American Girl" released by Paramount in 1929.

 His active years were from 1916 to 1933.
</item 8>

<item 9>
Jeff, One Lonely Guy
Jeff, One Lonely Guy is a 2012 nonfiction book by Jeff Ragsdale.

 It was published on March 20, 2012 by New Harvest.

 Dave Eggers selected the book for inclusion in The Best American Nonrequired Reading 2012, and it was a GQ 2012 "Book of the Year".

 In 2014 Amitava Kumar included portions of "Jeff, One Lonely Guy" in his newly released book, "A Matter of Rats: A Short Biography of Patna".

 Kumar previously interviewed Ragsdale and wrote about him in "The New York Times".
</item 9>

<item 10>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 10>
=== END ITEM id=d84ab90b77571b61 ===

=== ITEM id=b15d43ee4c4ba269 ===
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
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 1>

<item 2>
Ray Birdwhistell
Collaborations with others, including initially Margaret Mead and Gregory Bateson, and later, Erving Goffman and Dell Hymes had huge influence on his work.

 For example, the book he is best known for, "Kinesics and Context", "would not have appeared if it had not been envisaged by Erving Goffman" and he explicitly
stated "the paramount and sustaining influence upon my work has been that of anthropological linguistics", a tradition most directly represented at the
University of Pennsylvania by Hymes.
</item 2>

<item 3>
Erving Goffman
Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist
of the twentieth century".

 In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens,
Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
</item 3>

<item 4>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 4>

<item 5>
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
</item 5>

<item 6>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 6>

<item 7>
The Presentation of Self in Everyday Life
The Presentation of Self in Everyday Life is a 1956 sociology book by Erving Goffman, in which Goffman uses the imagery of the theatre in order to portray the
importance of human social interaction.

 Originally published in Scotland in 1956 and in the United States in 1959, it is Goffman’s first and most famous book, for which he received the American
Sociological Association’s MacIver award in 1961.
</item 7>

<item 8>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 8>

<item 9>
Goffman (disambiguation)
Erving Goffman (1922–1982) was a Canadian sociologist
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
=== END ITEM id=b15d43ee4c4ba269 ===

=== ITEM id=11fbe07c18b5b2ca ===
Gold sentences and facts:
Sentence 0:  He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink, T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
  Fact 0.0: This producer has produced songs for Justin Bieber.
  Fact 0.1: This producer has produced songs for Monica.
  Fact 0.2: This producer has produced songs for Britney Spears.
  Fact 0.3: This producer has produced songs for Usher.
  Fact 0.4: This producer has produced songs for Keri Hilson.
  Fact 0.5: This producer has produced songs for T.I.
  Fact 0.6: This producer has produced songs for Nelly Furtado.
  Fact 0.7: This producer has produced songs for Kevin Cossom.
  Fact 0.8: This producer has produced songs for Ciara.
  Fact 0.9: This producer has produced songs for Mariah Carey.
  Fact 0.10: This producer has produced songs for Timbaland.
  Fact 0.11: This producer has produced songs for Madonna.
  Fact 0.12: This producer has produced songs for J. Cole.
  Fact 0.13: This producer has produced songs for Kanye West.
  Fact 0.14: This producer has produced songs for Whitney Houston.
  Fact 0.15: This producer has produced songs for Missy Elliott.
  Fact 0.16: This producer has produced songs for M.I.A.
  Fact 0.17: This producer has produced songs for Justin Timberlake.
  Fact 0.18: This producer has produced songs for JoJo.
  Fact 0.19: This producer has produced songs for Joe Jonas.
  Fact 0.20: This producer has produced songs for Simple Plan.
  Fact 0.21: This producer has produced songs for The Clutch.
  Fact 0.22: This producer has produced songs for Pink.
  Fact 0.23: This producer has produced songs for T-Pain.
  Fact 0.24: This producer has produced songs for Diddy.
  Fact 0.25: This producer has produced songs for Meek Mill.
  Fact 0.26: This producer has produced songs for Bjork.
  Fact 0.27: This producer has produced songs for Duran Duran.
Sentence 1:  Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.
  Fact 1.0: Cole was raised in Fayetteville, North Carolina.
  Fact 1.1: Cole initially gained recognition as a rapper.
  Fact 1.2: Cole initially gained recognition as a rapper after the release of his debut mixtape.
  Fact 1.3: Cole's debut mixtape is titled The Come Up.
  Fact 1.4: Cole's debut mixtape The Come Up was released in early 2007.

Retrieved items:
<item 1>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 1>

<item 2>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 2>

<item 3>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 3>

<item 4>
Lil Ru
Sylvester Samuels better known by his stage name Lil Ru, is an American rapper from Ridgeway, South Carolina currently signed to Def Jam Recordings.

 His debut album, 21 & Up was released on August 25, 2009.

 The Ridgeway native was 16 when he made his professional foray into the music business.

 Inspired by New Orleans’ innovative Cash Money Crew, Ru began making a name for himself on his local music scene, doing live shows and pressing up his own CDs.

 His hard-hitting lyrics and entrepreneurial spirit caught the attention of fellow South Carolinian Angie Stone.

 Shortly after the neo-soul songstress helped him secure a deal with Elektra Records, Ru found himself unsigned again, among the artists lost in the shuffle
after the label merged with Atlantic Records.

 Music fans first heard him on his 2001 debut single Will Destroy.

 He then released his 2002 follow up, Shawty What You Doin’.

 Both songs reached the Billboard R&B/Hip-Hop charts and helped land him at his next label home, Capitol Records.
</item 4>

<item 5>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 5>

<item 6>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 6>

<item 7>
Anna Graceman
Anna Graceman is a young American singer-songwriter and pianist whose self-penned songs and music have gained attention at home and abroad.

 Her style of music incorporates elements of folk, soul and rock.

 As of 2016 she had written and performed over 70 of her own published songs; writing as well as producing her releases.

 Anna continues to divide her time between her solo projects and her new project - the ‘Graceman’ Band - a trio that includes Allie Graceman (Rhythm/Lead
Guitar) and Landon Graceman (Drums).
</item 7>

<item 8>
Raleigh Moncrief
Raleigh Moncrief is an American electronic musician, record producer and recording engineer who has worked with Zach Hill, Dirty Projectors and !!!

.
</item 8>

<item 9>
Daniel Boone Native Gardens
Daniel Boone Native Gardens, located in Boone, North Carolina, United States, has a collection of North Carolina native plants in an informal landscaped design.

 The gardens are open daily from May to October.
</item 9>

<item 10>
Kaydean
Kaydean is a British-American record and TV producer, composer, DJ, record label executive and artist manager.

 Kaydean produces, composes and arranges music in the Dance, Electronica, Latin Pop, Contemporary R&B, and Electronica genres.

 Born in London, he has worked with many artists including Janet Jackson, Maxi Priest, Robyn, Man Parrish, Angel "Cucco" Peña, Baron Lopez, Raekwon, Ky-Mani
Marley, Frankie Cutlass and Nocera amongst others.

 He is also known for discovering Latin Pop Star Janid and managing her career.

 He produced her Spanish album "La Magia" under the Sony Music Latin imprint label, Handshake and her Hit Single "Penicilina".

 In 2015 he debuted as a TV executive producer with Janid's reality show on Mega TV, The first episode aired on September 6, 2015.
</item 10>
=== END ITEM id=11fbe07c18b5b2ca ===

=== ITEM id=c2aec036ba77c74e ===
Gold sentences and facts:
Sentence 0: Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.
  Fact 0.0: Kim Dong-jun's Hangul name is 김동준.
  Fact 0.1: Kim Dong-jun's Hanja name is 金桐俊.
  Fact 0.2: Kim Dong-jun was born February 11, 1992.
  Fact 0.3: Kim Dong-jun is a South Korean singer.
  Fact 0.4: Kim Dong-jun is a South Korean actor.
Sentence 1: Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.
  Fact 1.0: Charlie Harper was born David Charles Perez.
  Fact 1.1: Charlie Harper was born on 25 May 1944.
  Fact 1.2: Charlie Harper was born in Hackney, London.
  Fact 1.3: Charlie Harper is a British singer.
  Fact 1.4: Charlie Harper is a British songwriter.

Retrieved items:
<item 1>
Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Super Junior-D&E, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 1>

<item 2>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 2>

<item 3>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 3>

<item 4>
Charlie Harper (singer)
Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.

 He is best known as the lead singer from the punk band U.K. Subs.
</item 4>

<item 5>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 5>

<item 6>
Kim Dong-jun (footballer)
Kim Dong-jun (Korean: 김동준 , born 19 December 1994) is a South Korean footballer who plays as a goalkeeper for Yonsei University and South Korea national team.
</item 6>

<item 7>
List of songs written by Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Donghae & Eunhyuk, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 7>

<item 8>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 8>

<item 9>
Kim Dong-jun
Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.

 He is a member of the South Korean boyband and its sub-group ZE:A Five and ZE:A J.
</item 9>

<item 10>
Collins &amp; Harlan
Collins & Harlan, the team of Arthur Collins and Byron G. Harlan formed a popular comic duet between 1903 and 1926.

 They sang ragtime standards as well as what were known as "Coon songs" – music sung by white performers in a black dialect.

 Their material also employed many other stereotypes of the time including Irishmen and farmers.

 Fellow recording artist Billy Murray nicknamed them "The Half-Ton Duo" as both men were rather overweight.

 Collins and Harlan produced many number one hits with recordings of minstrel songs such as "My Gal Irene", "I Know Dat I'll be Happy Til I Die", "Who Do You
Love?"

 and "Down Among the Sugarcane".

 Their song "That Funny Jas Band from Dixieland", recorded November 8, 1916, is among the first recorded uses of the word "jas" which eventually evolved to
"jass", and to the current spelling "jazz".
</item 10>
=== END ITEM id=c2aec036ba77c74e ===

=== ITEM id=1a85cba0276fc4bc ===
Gold sentences and facts:
Sentence 0: The Great War is a 26-episode documentary series from 1964 on the First World War.
  Fact 0.0: The Great War is a documentary series.
  Fact 0.1: The Great War has 26 episodes.
  Fact 0.2: The Great War is a documentary series from 1964.
  Fact 0.3: The Great War is about the First World War.
Sentence 1:  The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian Broadcasting Commission.
  Fact 1.0: The Great War was a co-production of the Imperial War Museum.
  Fact 1.1: The Great War was a co-production of the British Broadcasting Corporation.
  Fact 1.2: The Great War was a co-production of the Canadian Broadcasting Corporation.
  Fact 1.3: The Great War was a co-production of the Australian Broadcasting Commission.
Sentence 2:  The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.
  Fact 2.0: Michael Redgrave was the narrator of The Great War.
  Fact 2.1: Marius Goring gave readings in The Great War.
  Fact 2.2: Ralph Richardson gave readings in The Great War.
  Fact 2.3: Cyril Luckham gave readings in The Great War.
  Fact 2.4: Sebastian Shaw gave readings in The Great War.
  Fact 2.5: Emlyn Williams gave readings in The Great War.
Sentence 3: Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.
  Fact 3.0: Imperial War Museums (IWM) is a British national museum organisation.
  Fact 3.1: Imperial War Museums has branches at five locations in England.
  Fact 3.2: Three of the five Imperial War Museums branches are in London.

Retrieved items:
<item 1>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 1>

<item 2>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 2>

<item 3>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 3>

<item 4>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 4>

<item 5>
Black Country Living Museum
The Black Country Living Museum (formerly The Black Country Museum) is an open-air museum of rebuilt historic buildings in Dudley in the West Midlands of
England.

 It is located in the centre of the Black Country, 10 miles west of Birmingham.

 The museum occupies 105000 m2 of former industrial land partly reclaimed from a former railway goods yard, disused lime kilns, canal arm and former coal pits.
</item 5>

<item 6>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 6>

<item 7>
First Life (TV series)
First Life is a 2010 British nature documentary series written and presented by David Attenborough, also known by the expanded titles David Attenborough's First
Life (UK) and First Life with David Attenborough (USA).

 It was first broadcast in the USA as a two-hour special on the Discovery Channel on 24 October 2010.

 In the United Kingdom it was broadcast as a two-part series on BBC Two on 5 November 2010.

 "First Life" sees Attenborough tackle the subject of the origin of life on Earth.

 He investigates the evidence from the earliest fossils, which suggest that complex animals first appeared in the oceans around 540 million years ago, an event
known as the Cambrian Explosion.

 Trace fossils of multicellular organisms from an even earlier period, the Ediacaran biota, are also examined.

 Attenborough travels to Canada, Morocco and Australia, using some of the latest fossil discoveries and their nearest equivalents amongst living species to
reveal what life may have been like at that time.

 Visual effects and computer animation are used to reconstruct and animate the extinct life forms.
</item 7>

<item 8>
British National Bibliography
British National Bibliography (BNB) was established to publish a list of the books, journals and serials that are published in the United Kingdom and Republic
of Ireland.

 It also includes information on forthcoming titles.

 New books and serials have been recorded in the British National Bibliography (BNB) since 1950.

 The British National Bibliography was first established at the British Museum.

 This is the single most comprehensive listing of UK titles.

 UK and Irish publishers are obliged by legal deposit to send a copy of all new publications, including serial titles to BNB for listing.

 The BNB publish the list weekly in electronic form: the last printed weekly list appeared in December 2011.
</item 8>

<item 9>
The Great War (documentary)
The Great War is a 26-episode documentary series from 1964 on the First World War.

 The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian
Broadcasting Commission.

 The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.

 Each episode is  40 minutes long.
</item 9>

<item 10>
A Cream Cracker under the Settee
"A Cream Cracker Under The Settee" is a dramatic monologue written by Alan Bennett in 1987 for television, as part of his "Talking Heads" series for the BBC.

 The series became very popular, moving onto BBC Radio, international theatre, becoming one of the best-selling audio book releases of all time and included as
part of both the A-level and GCSE English syllabus.

 It was the sixth and final episode of the first series of "Talking Heads".
</item 10>
=== END ITEM id=1a85cba0276fc4bc ===
