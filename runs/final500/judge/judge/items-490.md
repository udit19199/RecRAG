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

=== ITEM id=5268c4e90632d649 ===
Gold sentences and facts:
Sentence 0: Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian, known for his surreal humour and art.
  Fact 0.0: Ash Lieb was born on 22 August 1982.
  Fact 0.1: Ash Lieb is an Australian artist.
  Fact 0.2: Ash Lieb is an Australian writer.
  Fact 0.3: Ash Lieb is an Australian comedian.
  Fact 0.4: Ash Lieb is known for his surreal humour.
  Fact 0.5: Ash Lieb is known for his surreal art.
Sentence 1: Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.
  Fact 1.0: Helen Elswyth Thane Ricker Beebe was born on May 16, 1900.
  Fact 1.1: Helen Elswyth Thane Ricker Beebe died on July 31, 1984.
  Fact 1.2: Helen Elswyth Thane Ricker Beebe was an American romance novelist.

Retrieved items:
<item 1>
Amaruk Kayshapanta: Ecuadorian multidisciplinary artist, immigrant-rights advocate, and Andean cultural ambassador and philosopher. Lorin Thompson: muralist and
Ranger Rick creator. Ron Bechet: New Orleans painter and longtime art chair. Hagop Sandaldjian: Armenian-American musician and microminiature sculptor. Philippe
Perrin: French artist and photographer. Richard Renaldi: American portrait photographer and photobook creator. Laurent Manrique: French chef-restaurateur whose
Aqua earned
</item 1>

<item 2>
Ash Lieb
Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian, known for his surreal humour and art.

 Born in Ballarat, Ash Lieb began exhibiting art at eight years of age, and at the age of fifteen, wrote his first novel, "The Secret Well".

 Throughout his career, Lieb has created a diverse range of artworks, books, short films, and comedic performances, which have often possessed philosophical or
psychiatric undertones.
</item 2>

<item 3>
Elswyth Thane
Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.

 Born in Burlington, Iowa, she was the daughter of a local teacher and high school principal.

 The family moved to New York City in 1918, and "Helen Ricker" changed her name to "Elswyth Thane".

 She began working as a freelance writer in the 20s, and became a newspaper writer and a Hollywood screenwriter.

 Her first novel, "Riders of the Wind", was published in 1926.

 Her novel, "The Tudor Wench", about Elizabeth I of England, was made into a play.

 She was a collector of scarves.
</item 3>

<item 4>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 4>

<item 5>
The entries profile artists, performers, filmmakers, musicians, writers and cultural topics. AES+F is a Russian art collective known for monumental video
installations; Lev Yilmaz created *Tales of Mere Existence*, while Clinton Walker chronicles Australian culture. Other figures include South African actor Milan
Murray; Swedish entertainer Henrik Dorsin; Japanese actor Bokuzen Hidari; Turkish electronic composer Erdem Helvacıoğlu; Lynch collaborator Mary Sweeney;
Icelandic actor Balt
</item 5>

<item 6>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 6>

<item 7>
The Preytells
The Preytells were an Australian indie rock band from Perth, formed in 2004 by Jessica Bennett (a.k.a. Audrey Tell) on lead guitar and backing vocals, Simon
Okely (a.k.a. Will Tell) on guitar and lead vocals, Jaclyn Pearson on drums and percussion and Cameron Stewart on bass guitar and backing vocals.

 Their sole album, "Flood Songs/June Songs", was issued in September 2009; before they disbanded later that year.

 Their single, "Shout!"

 ( May 2008), was nominated for WAM Song of the Year in the Pop category in 2008; while, "Lord Hold My Hand", was nominated for the same category in 2009.
</item 7>

<item 8>
The entries profile the Undying Ones’ two-headed demon leader, the Nameless One, and diverse cultural figures: novelist Leonard Cohen; writers Paul Scott, John
Eric Holmes, Douglas Hill, and Tatsuhiro Ōshiro; artists Ash Lieb and Scott Treleaven; translator William Scott Wilson; tailor Leonard Logsdail; and Kenyan
editor Billy Kahora. They also describe novels and literary projects, including House of Leaves, Simon Bloom, Tongan literature, and the Raj Quart
</item 8>

<item 9>
Tryst (novel)
Tryst, written in 1939 by Elswyth Thane, is a story of two people and a seemingly ordinary home.

 While a quick summary may make it sound like a Horror novel, it actually borders on Mystery and Romance.
</item 9>

<item 10>
Oliver Lieb
Oliver Lieb (born 1969 in Frankfurt, Germany) is a German electronic music producer and DJ.

 Lieb is known to have more than a dozen aliases with over 200 productions and remixes in various electronic genres such as trance, house, and techno.

 He has also produced tracks for other artists.
</item 10>
=== END ITEM id=5268c4e90632d649 ===

=== ITEM id=80ead24a81a42011 ===
Gold sentences and facts:
Sentence 0:  They were first year members of the Mountain West Conference.
  Fact 0.0: They were first-year members of the Mountain West Conference.
Sentence 1: The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I Football Bowl Subdivision (FBS) (formerly I-A).
  Fact 1.0: The Mountain West Conference (MW) is a collegiate athletic conference.
  Fact 1.1: The Mountain West Conference is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA).
  Fact 1.2: The Mountain West Conference is affiliated with the NCAA Division I Football Bowl Subdivision (FBS).
  Fact 1.3: The Football Bowl Subdivision (FBS) was formerly called I-A.

Retrieved items:
<item 1>
2012 Hawaii Warriors football team
The 2012 Hawaii Warriors football team represented the University of Hawaii at Manoa in the 2012 NCAA Division I FBS football season.

 They were led by first-year head coach Norm Chow and played their home games at Aloha Stadium.

 They were first year members of the Mountain West Conference.

 They finished the season 3–9, 1–7 in Mountain West play to finish in a tie for ninth place.
</item 1>

<item 2>
List of Pacific Tigers head football coaches
The Pacific Tigers college football team represented University of the Pacific (CA).

 The Tigers competed in the National Collegiate Athletic Association (NCAA) College Division in the years 1937–1968.

 In 1969, the team moved to National Collegiate Athletic Association (NCAA) Division I and stayed there until disbanding after the 1995 season.
</item 2>

<item 3>
2011–12 Wagner Seahawks men's basketball team
The 2011–12 Wagner Seahawks men's basketball team represented Wagner College during the 2011–12 NCAA Division I men's basketball season.

 The Seahawks, led by second year head coach Dan Hurley, played their home games at Spiro Sports Center and are members of the Northeast Conference.

 They finished the season 25–6, 15–3 in NEC play to finish in second place.

 They lost in the semifinals of the NEC Basketball Tournament to Robert Morris.

 Despite having 25 wins, the Seahawks did not accept an invitation to a post season tournament.
</item 3>

<item 4>
Men's National Collegiate Athletic Association
The Men's National Collegiate Athletic Association is an athletic association in the Philippines, and is the direct counterpart of the older Women's National
Collegiate Athletic Association (founded in 1970).

 The Men's National Collegiate Athletic Association (MNCAA) is exclusive for men.

 The MNCAA was founded in 2004.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 The competing sports of the tournament are Basketball, Volleyball, Futsal, Table Tennis, Badminton and Streetdance.
</item 4>

<item 5>
Women's National Collegiate Athletic Association
The Women's National Collegiate Athletic Association (WNCAA) is an athletic association in the Philippines exclusively for women.

 It was founded in 1970.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 Its men's counterpart is the Men's National Collegiate Athletic Association, founded in 2004.
</item 5>

<item 6>
2013–14 Iona Gaels men's basketball team
The 2013–14 Iona Gaels men's basketball team represented Iona College during the 2013–14 NCAA Division I men's basketball season.

 The Gaels, led by fourth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 22–11, 17–3 to win the MAAC regular season championship.

 They advanced to the championship game of the MAAC Tournament where they lost to Manhattan.

 As a regular season conference champion who failed to win their conference tournament, they received an automatic bid to the National Invitation Tournament
where they lost in the first round to Louisiana Tech.
</item 6>

<item 7>
Hawaii Rainbow Warriors baseball
The Hawaii Rainbow Warriors baseball team is a varsity intercollegiate athletic team of the University of Hawaii at Manoa in Honolulu, Hawaii, United States.

 The team is a member of the Big West Conference, which is part of the National Collegiate Athletic Association's Division I.

 Hawaii's first baseball team was fielded in 1923.

 The team plays its home games at Les Murakami Stadium in Honolulu, Hawaii.

 The Rainbow Warriors are coached by Mike Trapasso.
</item 7>

<item 8>
Mountain West Conference
The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I
Football Bowl Subdivision (FBS) (formerly I-A).

 The MW officially began operations in July 1999.

 Geographically, the MW covers a broad expanse of the Western United States, with member schools located in California, Colorado, Hawaii, Idaho, Nevada, New
Mexico, Utah, and Wyoming.

 Craig Thompson has served as Commissioner of the MW since its founding in 1999.
</item 8>

<item 9>
Wayne State Warriors football
The Wayne State Warriors football team is the college football team at Wayne State University in Detroit, Michigan.

 The Wayne State football team played their first game in October 1918.

 The Wayne State Warriors have competed in the Great Lakes Intercollegiate Athletic Conference since 1999 (and previously from 1975-1989), and are currently a
Division II member of the National Collegiate Athletic Association (NCAA).

 Wayne State plays their home games at Tom Adams Field at Wayne State Stadium.

 All Wayne State games are broadcast on WDTK radio.
</item 9>

<item 10>
2012–13 Wagner Seahawks men's basketball team
The 2012–13 Wagner Seahawks men's basketball team represented Wagner College during the 2012–13 NCAA Division I men's basketball season.

 The Seahawks were led by the youngest men's head coach in NCAA Division I, Bashir Mason, who was 28 when he was elevated from an assistant position in March
2012 following the departure of Dan Hurley for Rhode Island.

 The Seahawks played their home games at Spiro Sports Center and were members of the Northeast Conference.

 They finished the season 19–12, 12–6 in NEC play to finish in a three way tie for second place.

 They advanced to the semifinals of the Northeast Tournament where they lost to Long Island.

 For the second consecutive year, despite a winning record, Wagner choose not to participate in a post season tournament.
</item 10>
=== END ITEM id=80ead24a81a42011 ===

=== ITEM id=81348c957110d085 ===
Gold sentences and facts:
Sentence 0: Eugène Ionesco (born Eugen Ionescu, ] ; 26 November 1909 – 28 March 1994) was a Romanian-French playwright who wrote mostly in French, and one of the foremost figures of the French Avant-garde theatre.
  Fact 0.0: Eugène Ionesco was born Eugen Ionescu.
  Fact 0.1: Eugène Ionesco was born on 26 November 1909.
  Fact 0.2: Eugène Ionesco died on 28 March 1994.
  Fact 0.3: Eugène Ionesco was a Romanian-French playwright.
  Fact 0.4: Eugène Ionesco wrote mostly in French.
  Fact 0.5: Eugène Ionesco was one of the foremost figures of the French Avant-garde theatre.
Sentence 1: Paul Mark Scott (25 March 19201 March 1978) was an English novelist, playwright, and poet, best known for his monumental tetralogy "The Raj Quartet."
  Fact 1.0: Paul Mark Scott was born on 25 March 1920.
  Fact 1.1: Paul Mark Scott died on 1 March 1978.
  Fact 1.2: Paul Mark Scott was an English novelist.
  Fact 1.3: Paul Mark Scott was an English playwright.
  Fact 1.4: Paul Mark Scott was an English poet.
  Fact 1.5: Paul Mark Scott is best known for his monumental tetralogy "The Raj Quartet."

Retrieved items:
<item 1>
The Killer (play)
The Killer (French: Tueur sans gages , sometimes translated "The Killer without Reason" or "The Killer without Cause") is a play written by Eugène Ionesco in
1958.

 It is the first of Ionesco's Berenger plays, the others being "Rhinocéros" (1959), "Exit the King" (1962), and "A Stroll in the Air" (1963).
</item 1>

<item 2>
Paul Scott (novelist)
Paul Mark Scott (25 March 19201 March 1978) was an English novelist, playwright, and poet, best known for his monumental tetralogy "The Raj Quartet."

 His novel "Staying On" won the Booker Prize for 1977.
</item 2>

<item 3>
Lonely Planet (play)
Lonely Planet is a two character play written by Steven Dietz.

 The play tells the story of Jody and Carl, two gay men who live in an unnamed American city.

 The play was written during the midst of the AIDS epidemic, which is the central focus of the story, though ultimately the play sends the message that one
should pay attention to the world around them and realize its problems rather than shun it.

 The play heavily references the Eugène Ionesco comedy "The Chairs".
</item 3>

<item 4>
Eugène Ionesco
Eugène Ionesco (born Eugen Ionescu, ] ; 26 November 1909 – 28 March 1994) was a Romanian-French playwright who wrote mostly in French, and one of the foremost
figures of the French Avant-garde theatre.

 Beyond ridiculing the most banal situations, Ionesco's plays depict the solitude and insignificance of human existence in a tangible way.
</item 4>

<item 5>
The Lesson
The Lesson (French: "La Leçon" ) is a one-act play by French-Romanian playwright Eugène Ionesco.

 It was first performed in 1951 in a production directed by Marcel Cuvelier (who also played the Professor).

 Since 1957 it has been in permanent showing at Paris' Théâtre de la Huchette, on an Ionesco double-bill with The Bald Soprano.

 The play is regarded as an important work in the "Theatre of the Absurd".
</item 5>

<item 6>
Victims of Duty
Victims of Duty (French: Victimes du Devoir ) is a one-act play written in 1953 by French-Romanian playwright Eugène Ionesco.

 An early work, it has not received the notoriety of his other works.

 The play is in the Theatre of the Absurd style, of which Ionesco was a pioneer.
</item 6>

<item 7>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 7>

<item 8>
The New Tenant
The New Tenant (French: "Le Nouveau Locataire" ) is a play written by Eugène Ionesco in 1955.

 The central image is common to many Ionesco plays: something accumulates on stage and overwhelms the characters.

 In this case its furniture.

 The main characters are a gentleman, a caretaker, and two movers.

 The caretaker talks as the gentleman, the "new tenant" of the title, directs the two movers who continuously bring in furniture.
</item 8>

<item 9>
Exit the King
Exit the King (French: Le Roi se meurt ) is an absurdist drama by Eugène Ionesco that premiered in 1962.

 It is the third in Ionesco's "Berenger Cycle", preceded by "The Killer" (1958) and "Rhinocéros" (1959), and followed by "A Stroll in the Air" (1963).
</item 9>

<item 10>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 10>
=== END ITEM id=81348c957110d085 ===

=== ITEM id=8c55e6db6e01fbeb ===
Gold sentences and facts:
Sentence 0: And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
  Fact 0.0: And There's More is a comedy sketch show.
  Fact 0.1: And There's More stars Jimmy Cricket.
  Fact 0.2: And There's More was produced by Central Independent Television.
  Fact 0.3: And There's More was produced for ITV.
  Fact 0.4: And There's More was produced from 1985 till 1988.
Sentence 1:  He currently lives with his family in Rochdale, North West England.
  Fact 1.0: The person described in this passage currently lives with his family in Rochdale, North West England.
  Fact 1.1: The person described in this passage currently lives in Rochdale.
  Fact 1.2: Rochdale is in North West England.

Retrieved items:
<item 1>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 1>

<item 2>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 2>

<item 3>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 3>

<item 4>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 4>

<item 5>
Live in Old Smokey
Live in Old Smokey is a live CD by English singer/songwriter Linda Lewis recorded at Ronnie Scott's in London and issued in 2006.
</item 5>

<item 6>
Mazzy Star
Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and
bassist Kendra Smith.

 Roback's friend Hope Sandoval became the group's vocalist when Smith left the band.
</item 6>

<item 7>
Graph query row:
person: James Mulgrew
town: Rochdale
kind: None
country: None
</item 7>

<item 8>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 8>

<item 9>
Anastasia Horne
Anastasia Nicole "Ana" Horne (born August 5, 1978 in Claremont, California) is a former American actress and singer who has appeared on television and in the
theater.

 Her last television role was "Lori" on the MTV series "Undressed" in 2000.

 She also starred as "Lark Madison-Scanlon" on the daytime soap opera "Port Charles" from 1997 to 1999.

 As a teenager, she was cast as "Ana" on the long-running NBC/The Disney Channel series "Kids Incorporated" from 1984, to 1985, and 1991 to the end of the
series in 1994.
</item 9>

<item 10>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 10>
=== END ITEM id=8c55e6db6e01fbeb ===

=== ITEM id=99ebb8623f431281 ===
Gold sentences and facts:
Sentence 0: The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR: "Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet Union) and South Korea (with the principal support of the United States).
  Fact 0.0: The Korean War is called 한국전쟁 in South Korean Hangul.
  Fact 0.1: The Korean War is called 韓國戰爭 in South Korean Hanja.
  Fact 0.2: The Korean War is romanized as 'Hanguk Jeonjaeng' (RR), meaning 'Korean War'.
  Fact 0.3: The Korean War is called 조국해방전쟁 in North Korean Chosŏn'gŭl.
  Fact 0.4: The Korean War is called 祖國解放戰爭 in North Korean Hancha.
  Fact 0.5: The Korean War is romanized as 'Choguk haebang chǒnjaeng' (MR), meaning 'Fatherland Liberation War'.
  Fact 0.6: The Korean War lasted from 25 June 1950 to 27 July 1953.
  Fact 0.7: The Korean War was a war between North Korea and South Korea.
  Fact 0.8: North Korea fought with the support of China and the Soviet Union.
  Fact 0.9: South Korea fought with the principal support of the United States.
Sentence 1: The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.
  Fact 1.0: The Western Allied invasion of Germany was coordinated by the Western Allies.
  Fact 1.1: The Western Allied invasion of Germany took place during the final months of hostilities in the European theatre of World War II.
Sentence 2:  The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.
  Fact 2.0: The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945.
  Fact 2.1: After crossing the Rhine, the Western Allies fanned out and overran all of western Germany.
  Fact 2.2: The Western Allies overran western Germany from the Baltic in the north to Austria in the south.
  Fact 2.3: The Germans surrendered on 8 May 1945.

Retrieved items:
<item 1>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 1>

<item 2>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 2>

<item 3>
Allied invasion of Italy
The Allied invasion of Italy was the Allied amphibious landing on mainland Italy that took place on 3 September 1943 during the early stages of the Italian
Campaign of World War II.

 The operation was undertaken by General Sir Harold Alexander's 15th Army Group (comprising General Mark W. Clark and General Bernard Montgomery's British
Eighth Army) and followed the successful invasion of Sicily.

 The main invasion force landed around Salerno on 9 September on the western coast in Operation Avalanche, while two supporting operations took place in
Calabria (Operation Baytown) and Taranto (Operation Slapstick).
</item 3>

<item 4>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 4>

<item 5>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 5>

<item 6>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 6>

<item 7>
South German Offensive
The South German Offensive is the general name of one of the final offensives of World War II in Europe.

 The offensive was led by the Seventh and Third armies of the United States along with the First Army of France.

 Soviet troops linked up with American forces in Czechoslovakia notably in the Battle of Slivice.

 The offensive was made by the US 6th Army Group to protect the US 12th Army Group's right flank and to prevent a German last stand in the Alps.

 However German resistance was much more fierce than in the north, which slowed the 6th Army Group's progress.

 However, by the end of April, many German divisions surrendered without a fight to the advancing American forces to avoid the inevitable destruction.

 The VI Corps of the Seventh Army linked up with the US Fifth Army, which fought through Italy, in the Alps as the Third Army advanced into Austria and
Czechoslovakia, where it linked up with Soviet forces advancing from the east.

 Fighting continued a few days after the Surrender of Germany on 8 May, due to German forces fighting west to surrender to the Americans instead of the Soviets.
</item 7>

<item 8>
Schlieffen Plan
The Schlieffen Plan (German: "Schlieffen-Plan" , ] ) was the name given after World War I to the thinking behind the German invasion of France and Belgium on 4
August 1914.

 Field Marshal Alfred von Schlieffen, the Chief of the Imperial Army German General Staff from 1891 to 1906, devised a deployment plan for a war-winning
offensive, in a one-front war against the French Third Republic from 1905–06.

 After the war, the German official historians of the "Reichsarchiv" and other writers, described the plan as a blueprint for victory.

 German historians claimed that the plan had been ruined by "Generaloberst" (Colonel-General) Helmuth von Moltke the Younger, the Commander-in-Chief of the
German army after Schlieffen retired in 1906, who was dismissed after the First Battle of the Marne (5–12 September 1914).
</item 8>

<item 9>
Western Allied invasion of Germany
The Western Allied invasion of Germany was coordinated by the Western Allies during the final months of hostilities in the European theatre of World War II.

 The Allied invasion of Germany started with the Western Allies crossing the Rhine River in March 1945 before fanning out and overrunning all of western Germany
from the Baltic in the north to Austria in the south before the Germans surrendered on 8 May 1945.

 This is known as the "Central Europe Campaign" in United States military histories.
</item 9>

<item 10>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 10>
=== END ITEM id=99ebb8623f431281 ===

=== ITEM id=330bf96ac1789753 ===
Gold sentences and facts:
Sentence 0: The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.
  Fact 0.0: The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.
  Fact 0.1: Alberta is a Canadian province.
  Fact 0.2: Alberta is governed under fundamental rules described in the Constitution of Alberta.
Sentence 1:  As is typical of all Canadian provinces, and Westminster systems more generally, Alberta's is an unwritten constitution.
  Fact 1.0: Alberta's constitution is an unwritten constitution.
  Fact 1.1: It is typical of all Canadian provinces for their constitutions to be unwritten.
  Fact 1.2: It is typical of Westminster systems more generally for their constitutions to be unwritten.
Sentence 2: The Westminster system is a parliamentary system of government modelled after that which developed in the United Kingdom.
  Fact 2.0: The Westminster system is a parliamentary system of government.
  Fact 2.1: The Westminster system is modelled after the system of government that developed in the United Kingdom.

Retrieved items:
<item 1>
List of heavy metal festivals
This is an incomplete list of heavy metal festivals.

 Heavy metal (or simply metal) is a genre of rock music that developed in the late 1960s and early 1970s, largely in the United States and the United Kingdom.

 With roots in blues rock and psychedelic rock, the first heavy metal bands such as Led Zeppelin, Black Sabbath and Deep Purple attracted large audiences, and
during the late 1960s and mid-1970s these band and others in their genre were featured at a number of historic rock festivals. Judas Priest helped spur the
genre's evolution by discarding much of its blues influence.
</item 1>

<item 2>
Governance of England
There has not been a government of England since 1707 when the Kingdom of England ceased to exist as a sovereign state, as it merged with the Kingdom of
Scotland to form the Kingdom of Great Britain.

 Kingdom of Great Britain continued from 1707 until 1801 when it merged with the Kingdom of Ireland to form the United Kingdom of Great Britain and Ireland,
which itself became the United Kingdom of Great Britain and Northern Ireland (UK) in 1922 (in reality; in name in 1927) upon independence for most of the island
of Ireland.

 The UK since then has gone through significant change to its system of government, with devolved parliaments, assemblies and governments in Scotland, Wales and
Northern Ireland.

 England, however, remains under the full jurisdiction, on all matters, of the Parliament of the United Kingdom of Great Britain and Northern Ireland and the UK
government as no devolved administration has been created for England within the new structure.

 This situation has led to the anomaly, known as the West Lothian question, which is that Scottish Members of Parliament (MPs) are able to vote on legislation
that affects only England whereas English MPs can not vote on certain Scottish matters due to devolution.
</item 2>

<item 3>
Australian Public Service
The Australian Public Service (APS) is the federal civil service of the Commonwealth of Australia responsible for the public administration, public policy, and
public services of the departments and executive and statutory agencies of the Government of Australia.

 The Australian Public Service was established at the Federation of Australia in 1901 as the Commonwealth Public Service and modeled on the Westminster system
and United Kingdom's Civil Service.

 The establishment and operation of the Australian Public Service is governed by the "Public Service Act 1999" of the Parliament of Australia as an "apolitical
public service that is efficient and effective in serving the Government, the Parliament and the Australian public".

 The conduct of Australian public servants is also governed by a Code of Conduct and guided by the APS Values set by the Australian Public Service Commission.
</item 3>

<item 4>
British North America Acts
The British North America Acts 1867–1975 are the original names of a series of Acts at the core of the constitution of Canada.

 They were enacted by the Parliament of the United Kingdom and the Parliament of Canada.

 In Canada, some of the Acts were amended or repealed by the Constitution Act, 1982.

 The rest were renamed in Canada as the "Constitution Acts".

 In the United Kingdom, those Acts that were passed by the British Parliament remain under their original names.

 The term "British North America" (BNA) refers to the British colonies in North America.
</item 4>

<item 5>
Westminster system
The Westminster system is a parliamentary system of government modelled after that which developed in the United Kingdom.

 This term comes from the Palace of Westminster, the seat of the British parliament.
</item 5>

<item 6>
Constitution of Alberta
The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.

 As is typical of all Canadian provinces, and Westminster systems more generally, Alberta's is an unwritten constitution.

 Alberta's constitution, like Britain's (on which it is modeled), includes any and all pieces of legislation, court decisions, proclamations, and conventions
which together inform how the province operates.

 Many statutes are important to understanding the governance of the province, but nowhere are they consolidated into a single document or even a list.

 The office of Attorney-General at one time suggested 23 acts which might be included, but cautioned that this was not a "definitive list".

 However, since Alberta is a part of federation, its powers are clearly delineated in law, via the Constitution of Canada.
</item 6>

<item 7>
Politics of Canada
The politics of Canada function within a framework of parliamentary democracy and a federal system of parliamentary government with strong democratic
traditions.

 Canada is a constitutional monarchy, in which the Monarch is head of state.

 The country has a multi-party system in which many of its legislative practices derive from the unwritten conventions of and precedents set by the United
Kingdom's Westminster Parliament.

 However, Canada has evolved variations: party discipline in Canada is stronger than in the United Kingdom and more parliamentary votes are considered motions
of confidence, which tends to diminish the role of non-Cabinet Members of Parliament, (MPs).

 Such members, in the government caucus, and junior or lower-profile members of opposition caucuses, are known as backbenchers. Backbenchers can, however, exert
their influence by sitting in parliamentary committees, like the Public Accounts Committee or the National-Defence Committee.
</item 7>

<item 8>
Quebec Legislature
The Quebec Legislature (officially Parliament of Quebec, French: "Parlement du Québec" ) is the legislature of the province of Quebec, Canada, since
Confederation in 1867.

 Today, the legislature is made of two elements: the monarch of Canada, represented by the Lieutenant Governor of Quebec, and the unicameral assembly called the
National Assembly of Quebec.

 The legislature has existed since 1867 when Quebec, then called Canada East, became one of the founding colonies of Canadian Confederation.

 From 1867 until 1968 the legislature was bicameral, containing a lower chamber called the Legislative Assembly of Quebec and an upper chamber called the
Legislative Council of Quebec.
</item 8>

<item 9>
Canada under British rule
Canada first came under British rule with the Treaty of Paris (1763) which ceded New France, of which Canada was a part, to the British Empire.

 Gradually, other territories, colonies, and provinces that were part of British North America would be added to Canada.

 The Royal Proclamation of 1763 enlarged the colony of Canada under the name of the Province of Quebec, which with the Constitutional Act 1791 became known as
The Canadas.

 With the Act of Union 1840 Upper and Lower Canada were joined to become the United Province of Canada.

 Later, with Confederation in 1867, the British maritime colonies of New Brunswick and Nova Scotia were joined with the British colony of Canada to form the
Dominion of Canada, which was subsequently divided into four provinces, Ontario, Quebec, New Brunswick, and Nova Scotia.

 A number of other British colonies, such as Newfoundland and British Columbia, and large territories such as Rupert's Land initially remained outside of the
newly formed federation.

 Over time, the remaining colonies and territories within British North America came under the control of Canada until the current geographic extent of the
country was reached when Newfoundland and Labrador joined Canada in 1949.
</item 9>

<item 10>
Canada under British rule
Although confederation in 1867 led to an enlarged Dominion with increased autonomy over domestic affairs, Canada still remained a colony within the British
Empire and was thus subordinate to the British Parliament until the enactment of the Statute of Westminster in 1931.

 This statute recognized Canada as an independent peer coequal with the United Kingdom, and thus provided the Parliament of Canada with legislative sovereignty
over all federal matters except the power to change the constitutional laws of Canada which remained under the purview of the Parliament of the United Kingdom.

 Canada's final vestige of legal dependence on the United Kingdom was terminated in 1982 with the enactment of the Canada Act, subsequently providing Canada
with full legal sovereignty completely independent of the United Kingdom.
</item 10>
=== END ITEM id=330bf96ac1789753 ===

=== ITEM id=053a7d5930dbb881 ===
Gold sentences and facts:
Sentence 0: Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.
  Fact 0.0: Ted Kooshian was born October 8, 1961.
  Fact 0.1: Ted Kooshian is an American jazz pianist.
  Fact 0.2: Ted Kooshian is an American jazz keyboardist.
  Fact 0.3: Ted Kooshian is an American jazz composer.
  Fact 0.4: Ted Kooshian has performed with Aretha Franklin.
  Fact 0.5: Ted Kooshian has performed with Chuck Berry.
  Fact 0.6: Ted Kooshian has performed with Marvin Hamlisch.
  Fact 0.7: Ted Kooshian has performed with Sarah Brightman.
  Fact 0.8: Ted Kooshian has performed with Il Divo.
  Fact 0.9: Ted Kooshian has performed with Blood, Sweat, and Tears.
Sentence 1:  Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
  Fact 1.0: Marvin Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
Sentence 2:  This collection of all four is referred to as an "EGOT".
  Fact 2.0: The collection of all four awards (Emmy, Grammy, Oscar and Tony) is referred to as an 'EGOT'.

Retrieved items:
<item 1>
Leonard Cohen
Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and
painter.

 His work explored religion, politics, isolation, sexuality, and personal relationships.

 Cohen was inducted into the Canadian Music Hall of Fame, the Canadian Songwriters Hall of Fame, and the Rock and Roll Hall of Fame.

 He was a Companion of the Order of Canada, the nation's highest civilian honor.

 In 2011, Cohen received one of the Prince of Asturias Awards for literature and the ninth Glenn Gould Prize.
</item 1>

<item 2>
Marvin Hamlisch
Marvin Frederick Hamlisch (June 2, 1944August 6, 2012) was an American composer and conductor.

 Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.

 This collection of all four is referred to as an "EGOT".

 He is one of only two people (along with Richard Rodgers) to have won those four prizes and a Pulitzer Prize.
</item 2>

<item 3>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 3>

<item 4>
Philip K. Dick Award
The Philip K. Dick Award is a science fiction award given annually at Norwescon sponsored by the Philadelphia Science Fiction Society and (since 2005) supported
by the Philip K. Dick Trust, and named after science fiction and fantasy writer Philip K. Dick.

 It has been awarded since 1983, the year after Dick's death.

 Works that have received the award are identified on their covers as "Best Original SF Paperback".

 They are awarded to the best original paperback published each year in the US.
</item 4>

<item 5>
Toshi (musician)
Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and
songwriter.

 He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.

 Toshi then went on to have an extensive solo career.

 In 2007, X Japan reunited and began an ongoing world tour in 2010.

 His distinctive, impassioned tenor vocals have been compared to those of Steve Perry.
</item 5>

<item 6>
Tatsuhiro Oshiro
Tatsuhiro Ōshiro (大城 立裕 , Ōshiro Tatsuhiro , born 19 September 1925) is a Japanese novelist and playwright.

 He was awarded the Akutagawa Prize in 1967 for his novella of the same year, "The Cocktail Party", which has been adapted for the stage and made into a film.

 Ōshiro has also been an innovator of the traditional Ryukyuan narrative dance form known as kumi odori.

 Having added twenty new pieces to the repertoire, Ōshiro is credited as having "single-handedly revived the genre that originated in the 18th century" by
incorporating Okinawa shibai (dramas in the Okinawan language) and distinctive rhythms to construct a fluid, hybrid cultural identity.

 His writings have been noted for making Okinawan culture and history accessible to Japanese readership, while his more popular works have been critically
praised for "offering an acute perspective on the psychological and moral implications of war and military occupation."
</item 6>

<item 7>
List of James Tiptree Jr. Award winners
The following is a list of winners and shortlisted works of the James Tiptree Jr.

 Award, an annual literary prize for works of science fiction or fantasy that expand or explore one's understanding of gender.

 It was initiated in February 1991 by science fiction authors Pat Murphy and Karen Joy Fowler, and is awarded and discussed yearly at WisCon.
</item 7>

<item 8>
Triple Crown of Acting
The Triple Crown of Acting is a term used in the American entertainment industry to describe actors who have won a competitive Academy Award, Emmy Award, and
Tony Award in the acting categories.

 23 people have achieved the triple crown of acting (14 women, 9 men).

 Helen Hayes' Emmy Award win on February 5, 1953, made her the first person to achieve the triple crown.

 Thomas Mitchell became the first man to achieve the triple crown with his Tony Award win later the same year on March 29, 1953.

 Hayes and Rita Moreno are the only triple crown winners in competitive acting categories who have also won a Grammy Award to complete the EGOT.
</item 8>

<item 9>
List of awards and nominations received by Kim Tae-yeon
List of awards and nominations received by Kim Tae-yeon
</item 9>

<item 10>
Lonesome Pine Fiddlers
This most talented family of musicians were the best, surpassed by none.

 Finally, on October 1, 2009, The Lonesome Pine Fiddlers got their due when they were inducted into the International Bluegrass Hall of Fame at the Ryman
Theater (the original home of the Grand Ole Opry).

 Bobby Osborne, Melvin Goins and Paul Williams were there to receive the bands award.

 In the crowd of a sold out theater was the son of Ezra Cline, Scotty Ireland Cline, who recalled being in that same theater as a child sitting on stage and
watching the Fiddlers play.

 (At the time, the Opry had bleachers for family just off stage).

 The final act of the evening at the IBMA Awards was the playing of "Pain in my Heart" by Osborne, Goins and Williams along with a Song from the Dillards, who
were also inducted the same evening.
</item 10>
=== END ITEM id=053a7d5930dbb881 ===

=== ITEM id=8bb07ec75661df4e ===
Gold sentences and facts:
Sentence 0:  He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and the Highest Distinction of the Argentine Air Force.
  Fact 0.0: He was awarded the highest national military decoration, the Argentine Nation to the Heroic Valour in Combat Cross.
  Fact 0.1: He was awarded the Argentine Congressional Medal.
  Fact 0.2: He was awarded the Highest Distinction of the Argentine Air Force.
Sentence 1:  In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
  Fact 1.0: In 2010, the entity referred to here had 14,600 military personnel.
  Fact 1.1: In 2010, the entity referred to here had 6,900 civilian personnel.

Retrieved items:
<item 1>
South America air forces maneuvers
The South American air forces performs several joint aerial combat training exercises.

 Among the more important are Cruzex ( Portuguese: "Exercício Cruzeiro do Sul" ) which is hosted by the Brazilian Air Force, Salitre ( English: Saltpeter )
hosted by the Chilean Air Force and Ceibo ( English: Erythrina crista-galli flower ) hosted by the Argentine Air Force.

 The goal is to train together in order to respond to a crisis or integrate into United Nations peacekeeping operations as a unified team; the United States Air
Force and the Royal Canadian Air Force have also participated in some of these exercises in recent years.
</item 1>

<item 2>
Servicio de Inteligencia de la Fuerza Aérea
Servicio de Inteligencia de la Fuerza Aérea ("Air Force Intelligence Service", SIFA) is the intelligence agency of the Argentine Air Force.

 It is part of J-2.

 Its director is Commodore García.
</item 2>

<item 3>
Argentine Air Force Mobile Field Hospital
The Argentine Air Force Mobile Field Hospital (Spanish: "Hospital Reubicable de la Fuerza Aérea" ) is a field hospital operated by the Argentine Air Force.

 It is one of three health centers of its kind worldwide.
</item 3>

<item 4>
Cuerpo de Fuerzas Especiales
The Mexican Cuerpo de Fuerzas Especiales (Special Forces Corps) is a special forces unit of the Mexican Army.

 Formerly the "GAFE" (Grupo Aeromóvil de Fuerzas Especiales | Special-Forces Airmobile Group), the SF Corps has six battalions; one is the "Fuerza especial de
reaccion", a quick-response unit, and one is assigned to the Paratroopers Rifle Brigade; the motto of the SF Corps is "Todo por México" (Everything for Mexico).

 Within the SF Corps, there are regular, intermediate, and veteran -service troops.

 The regular-service soldiers usually operate as light infantry.

 The intermediate-service soldiers (lieutenants and captains) usually are instructors.

 The veteran-service soldiers of the Grupo Aeromóvil de Fuerzas Especiales del Alto Mando (GAFE High Command) handle Black-Ops missions.

 Also known as the COIFE, the Special Forces Corps of the Mexican Army is equivalent to the U.S. Army Special Forces.
</item 4>

<item 5>
Argentine Air Force
The Argentine Air Force (Spanish: Fuerza Aérea Argentina , or simply FAA) is the national aviation branch of the Armed Forces of the Argentine Republic.

 In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
</item 5>

<item 6>
Hector Fautario
Hector Luis Fautario was a retired Argentine Air Force general.

 He was the General Commander of the Argentine Air Force from 1973 to 1975.

 Shortly prior to the 1976 Argentine coup d'état, he was the government's only remaining loyal officer.

 He had drawn harsh criticism towards himself from the Army and Navy due to his vehement opposition to their repressive plans, and for his refusal to mobilize
the Air Force against the guerrillas' strongholds in the north.

 He was General Jorge Rafael Videla's last obstacle on the way to power.
</item 6>

<item 7>
Pablo Carballo
Commodore Pablo Marcos Rafael Carballo (b. 11 December 1947 Buenos Aires) is a retired member of the Argentine Air Force - the "Fuerza Aérea Argentina" (FAA) -
who fought in the 1982 Falklands War ( Spanish: "Guerra de las Malvinas" ) where he participated in actions that led to the sinking of three Royal Navy ships.

 He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and
the Highest Distinction of the Argentine Air Force.
</item 7>

<item 8>
Special Forces Regiment (Airborne)
The Special Forces Regiment (Airborne) is a special forces unit of the Philippine Army and one of three specialized regiments of the Special Operations Command.

 The unit is based on and continually trains with its american counterpart, the U.S. Army Special Forces (Green Berets).
</item 8>

<item 9>
Teodoro Waldner
Teodoro Guillermo Waldner (born 1927) is a retired Argentine Air Force commander who served as Air Force Chief of the General Staff from 1983 to 1985.
</item 9>

<item 10>
Cicaré CK.1
The Cicaré CK.1 (originally, the CH.III Colibrí) was a light helicopter developed in Argentina in the 1970s.

 It was a small, single-rotor aircraft of pod-and-boom configuration with a fully enclosed bubble canopy that could seat three people side-by-side.

 Cicaré's previous helicopter designs had attracted the attention of the Argentine Air Force, which in 1974 contracted him to develop a light helicopter for
training and also marketed for agricultural use.

 A prototype, registered "LV-X62" flew in September 1976, and the Air Force placed an order for five pre-production machines.

 However, development was terminated at this point.
</item 10>
=== END ITEM id=8bb07ec75661df4e ===

=== ITEM id=3c90ec8580d49459 ===
Gold sentences and facts:
Sentence 0: Fastbacks were a Seattle, Washington, punk rock band.
  Fact 0.0: Fastbacks were a punk rock band.
  Fact 0.1: Fastbacks were from Seattle, Washington.
Sentence 1: Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.
  Fact 1.0: Local Natives is an American indie rock band.
  Fact 1.1: Local Natives was previously known as Cavil at Rest.
  Fact 1.2: Local Natives is based in Silver Lake, Los Angeles, California, United States.

Retrieved items:
<item 1>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 1>

<item 2>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 2>

<item 3>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 3>

<item 4>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 4>

<item 5>
Fastbacks
Fastbacks were a Seattle, Washington, punk rock band.

 Formed in 1979 by songwriter/guitarist Kurt Bloch (born August 28, 1960), and friends Lulu Gargiulo (guitar and vocals, born October 12, 1960) and Kim Warnick
(bass and vocals, born April 7, 1959), they disbanded in 2001.

 Their sound mixed a generally punk rock approach to vocals and sound textures with poppy tunes and strong musicianship.
</item 5>

<item 6>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 6>

<item 7>
Hummingbird (Local Natives album)
Hummingbird is the second studio album by American indie rock band Local Natives, released on January 29, 2013 on Frenchkiss Records in the US, and on
Infectious Records in Europe.

 Produced by The National guitarist Aaron Dessner, the album was preceded by the singles, "Breakers" and "Heavy Feet."

 "Breakers" reached #12 on the Billboard Top 200.

 The song "Mt. Washington" was also featured in the 2015 video game "Life Is Strange".
</item 7>

<item 8>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 8>

<item 9>
Back to the Beat (EP)
Back to the Beat was the first significant EP release by the Minneapolis based pop punk band Motion City Soundtrack.

 The EP was recorded in Sound In Motion Recordings and released by Modern Radio Records and includes tracks that were later re-released.

 "Throw Down", "Back to the Beat" and "Capital H" were re-released on a split EP with Schatzi.

 "Capital H" was re-released on Motion City Soundtrack's first album "I Am the Movie".

 At the time of recording, the band members differed from the Motion City Soundtrack line-up formed in 2002.

 They were Joshua Cain on guitar and vocals, Austin Lindstrom on bass guitar, Joel Habedank on drums and Justin Pierre on lead vocals, guitar and keyboard.
</item 9>

<item 10>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 10>
=== END ITEM id=3c90ec8580d49459 ===

=== ITEM id=0ff22edf8292d692 ===
Gold sentences and facts:
Sentence 0: Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.
  Fact 0.0: Enrico Cocozza was born on 6 November 1921.
  Fact 0.1: Enrico Cocozza died on 27 December 1997.
  Fact 0.2: Enrico Cocozza was a Scottish filmmaker.
  Fact 0.3: Enrico Cocozza won many film awards during the 1940s and 1950s.
Sentence 1: ", which was nominated for the Academy Award for Best Foreign Language Film.
  Fact 1.0: A film was nominated for the Academy Award for Best Foreign Language Film.
  Fact 1.1: The film whose title is quoted in this fragment was nominated for the Academy Award for Best Foreign Language Film.

Retrieved items:
<item 1>
Academy Award for Best Film Editing
The Academy Award for Best Film Editing is one of the annual awards of the Academy of Motion Picture Arts and Sciences (AMPAS).

 Nominations for this award are closely correlated with the Academy Award for Best Picture.

 For 33 consecutive years, 1981 to 2013, every Best Picture winner had also been nominated for the Film Editing Oscar, and about two thirds of the Best Picture
winners have also won for Film Editing.

 Only the principal, "above the line" editor(s) as listed in the film's credits are named on the award; additional editors, supervising editors, etc. are not
currently eligible.

 The nominations for this Academy Award are determined by a ballot of the voting members of the Editing Branch of the Academy; there were 220 members of the
Editing Branch in 2012.

 The members may vote for up to five of the eligible films in the order of their preference; the five films with the largest vote totals are selected as
nominees.

 The Academy Award itself is selected from the nominated films by a subsequent ballot of all active and life members of the Academy.
</item 1>

<item 2>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 2>

<item 3>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 3>

<item 4>
Salaam Bombay!
Salaam Bombay!

 is a 1988 Hindi film directed by Mira Nair, and screenwritten by her longtime creative collaborator, Sooni Taraporevala.

 The film chronicles the day-to-day life of children living on the streets of Bombay, India's biggest city.

 It won the National Film Award for Best Feature Film in Hindi, the National Board of Review Award for Top Foreign Film, the Golden Camera and Audience Awards
at the Cannes Film Festival, and three awards at the Montréal World Film Festival.

 The film was India's second film submission to be nominated for the Academy Award for Best Foreign Language Film.

 The film was among the list of "The Best 1,000 Movies Ever Made" by "The New York Times".
</item 4>

<item 5>
Maya Deren Award
The American Film Institute Award for Independent Film and Video Artists, subtitled and generally known as the Maya Deren Award, was an award presented to
filmmakers and video artists by the American Film Institute to honor independent filmmaking.

 Named for the avant-garde experimental film artist Maya Deren, it was given from 1986 through 1996.
</item 5>

<item 6>
Enrico Cocozza
Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.

 His often surreal films were mainly filmed in and around the town of Wishaw in Scotland, where his family owned the popular Belhaven Cafe.

 These include "Chick's Day" (1950), a prize winner at the 1951 Scottish Amateur Film Festival, "The Living Ghost" (1957), and "Glasgow's Docklands" (1959).

 Illness later forced him to give up making films and he spent most of his working life teaching at the University of Strathclyde in Glasgow.

 In 2001 he was the subject of the Channel 4 documentary "Artery: the Story of Enrico Cocozza".
</item 6>

<item 7>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 7>

<item 8>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 8>

<item 9>
Two Brides and a Baby
Two Brides and a Baby is a 2011 Nigerian romantic drama film directed by Teco Benson, starring Keira Hewatch, Kalu Ikeagwu, OC Ukeje, Chelsea Eze, Stella
Damasus-Aboderin and Okey Uzoeshi.

 It premiered on November 17, 2011.

 It received awards and nominations at Africa Movie Academy Awards, Best of Nollywood Awards and Africa Magic Viewers Choice Awards.
</item 9>

<item 10>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 10>
=== END ITEM id=0ff22edf8292d692 ===

=== ITEM id=9fa73aa07d1bb029 ===
Gold sentences and facts:
Sentence 0:  The film stars Juno Temple, Julia Garner, Alessandro Nivola, Maggie Siff, Olivia Luccardi and Philip Ettinger.
  Fact 0.0: The film stars Juno Temple.
  Fact 0.1: The film stars Julia Garner.
  Fact 0.2: The film stars Alessandro Nivola.
  Fact 0.3: The film stars Maggie Siff.
  Fact 0.4: The film stars Olivia Luccardi.
  Fact 0.5: The film stars Philip Ettinger.
Sentence 1:  She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight Rises".
  Fact 1.0: The actress has appeared in the film Black Mass.
  Fact 1.1: The actress has appeared in the film The Other Boleyn Girl.
  Fact 1.2: The actress has appeared in the film Wild Child.
  Fact 1.3: The actress has appeared in the film Atonement.
  Fact 1.4: The actress has appeared in the film Maleficent.
  Fact 1.5: The actress has appeared in the film The Three Musketeers.
  Fact 1.6: The actress has appeared in the film The Dark Knight Rises.

Retrieved items:
<item 1>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 1>

<item 2>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 2>

<item 3>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 3>

<item 4>
Lovelace (film)
Lovelace is a 2013 American biographical drama film about porn actress Linda Boreman, better known as Linda Lovelace, star of "Deep Throat", a seminal 1972 film
at the forefront of the Golden Age of Porn.

 "Lovelace" covers her life from age 20 to 32.

 Directed by Rob Epstein and Jeffrey Friedman, the film was written by Andy Bellin and stars Amanda Seyfried, Peter Sarsgaard, Sharon Stone, Adam Brody, James
Franco, Chloë Sevigny, and Juno Temple.

 The film had its world premiere on January 22, 2013, at the 2013 Sundance Film Festival and opened in a U.S. limited release on August 9, 2013.
</item 4>

<item 5>
Cracks (film)
Cracks is a 2009 independent drama psychological thriller film starring Eva Green, James McGovern, Juno Temple, María Valverde, and Imogen Poots, which was
released theatrically in the UK and Ireland on 4 December 2009.

 In the United States it was released by IFC Films on 18 March 2011, and premiered on Showtime in late 2011.
</item 5>

<item 6>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 6>

<item 7>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 7>

<item 8>
Juliet Weber
Juliet Weber is a producer, director, writer, and an editor for the film industry.

 She co-produced films such as Fastpitch (2000) and Unchained Memories (2003).
</item 8>

<item 9>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 9>

<item 10>
Dirty Girl (2010 film)
Dirty Girl is a 2010 American coming of age comedy-drama film written and directed by Abe Sylvia.

 It stars Juno Temple, Milla Jovovich and William H. Macy.

 It premiered at the 2010 Toronto International Film Festival on September 12, 2010.

 It was distributed theatrically by The Weinstein Company on October 7, 2011.
</item 10>
=== END ITEM id=9fa73aa07d1bb029 ===
