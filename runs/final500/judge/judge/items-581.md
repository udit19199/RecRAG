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

=== ITEM id=0d5e4593a98b33d6 ===
Gold sentences and facts:
Sentence 0: Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.
  Fact 0.0: Heavenly Creatures is a 1994 New Zealand psychological drama.
  Fact 0.1: Heavenly Creatures was directed by Peter Jackson.
  Fact 0.2: Peter Jackson co-wrote the screenplay with his partner, Fran Walsh.
  Fact 0.3: Heavenly Creatures is about the notorious 1954 Parker-Hulme murder case.
  Fact 0.4: The Parker-Hulme murder case took place in Christchurch, New Zealand.
Sentence 1:  The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.
  Fact 1.0: The main premise of Heavenly Creatures deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme.
  Fact 1.1: Pauline Parker and Juliet Hulme murder Pauline Parker's mother.
Sentence 2: Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk series.
  Fact 2.0: Anne Perry was born on 28 October 1938 as Juliet Marion Hulme.
  Fact 2.1: Anne Perry is an English author of historical detective fiction.
  Fact 2.2: Anne Perry is best known for her Thomas Pitt and William Monk series.

Retrieved items:
<item 1>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 1>

<item 2>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 2>

<item 3>
Heavenly Creatures
Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the
notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.

 The film features Melanie Lynskey and Kate Winslet in their screen debuts with supporting roles by Sarah Peirse, Diana Kent, Clive Merrison, and Simon
O'Connor.

 The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.

 The events of the film cover the period from their meeting in 1952 to the murder in 1954.
</item 3>

<item 4>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 4>

<item 5>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 5>

<item 6>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 6>

<item 7>
Parker–Hulme murder case
The Parker–Hulme murder case began in the city of Christchurch, New Zealand, on 22 June 1954, when Honora Rieper (also known as Honora Parker, her legal name)
was killed by her teenaged daughter, Pauline Parker, and Pauline's close friend Juliet Hulme (later known as Anne Perry).

 Parker was 16 at the time, while Hulme was 15.
</item 7>

<item 8>
Hulder
A hulder is a seductive forest creature found in Scandinavian folklore.

 (Her name derives from a root meaning "covered" or "secret".)

 In Norwegian folklore, she is known as huldra ("the [archetypal] hulder", though folklore presupposes that there is an entire Hulder race and not just a single
individual).

 She is known as the skogsrå "forest spirit" or Tallemaja "pine tree Mary" in Swedish folklore, and ulda in Sámi folklore.

 Her name suggests that she is originally the same being as the völva divine figure Huld and the German Holda.
</item 8>

<item 9>
The Bone People
The Bone People (styled by the writer and in some editions as the bone people) is a Booker Prize-winning 1984 novel by New Zealand writer Keri Hulme.
</item 9>

<item 10>
Peter Elliott (actor)
Peter Elliott is a leading New Zealand actor.

 He has appeared in numerous television shows including "Shortland Street", "Gloss" and .

 He has also appeared in several movies including "Heavenly Creatures".

 Peter has a daughter Lucy Elliott who is an actress, playing character Dayna Jenkins on "Shortland Street" in 2013.
</item 10>
=== END ITEM id=0d5e4593a98b33d6 ===

=== ITEM id=cb916c1df3ded8ad ===
Gold sentences and facts:
Sentence 0:  Her favorite events are 800 metres and 1500 metres.
  Fact 0.0: This person's favorite events are the 800 metres.
  Fact 0.1: This person's favorite events are the 1500 metres.
Sentence 1:  The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.
  Fact 1.0: The distance has been contested at the Summer Olympics since 1896.
  Fact 1.1: The distance has been contested at the World Championships in Athletics since 1983.

Retrieved items:
<item 1>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 1>

<item 2>
1500 metres
The 1500 metres or 1,500-metre run (typically pronounced 'fifteen-hundred metres') is the foremost middle distance track event in athletics.

 The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.

 It is equivalent to 1.5 kilometers or approximately ⁄  miles.
</item 2>

<item 3>
Siegmar Ohlemann
Siegmar ("Sig") Ohlemann (born May 11, 1938) is a retired middle distance runner, who represented Canada in three events (800 metres, 4 × 100 metres and 4 × 400
metres) at the 1960 Summer Olympics in Rome, Italy.

 He claimed the silver medal in 800 metres at the 1963 Pan American Games in Brazil, behind fellow Canadian Don Bertoia.
</item 3>

<item 4>
K. M. Binu
Kalayathumkuzhi Mathews Binu (born 20 December 1980) is an Indian track and field athlete from Kerala who specializes in 400 metres and 800 metres.

 He holds the current 400 metres national record of 45.48 s set at the 2004 Athens Olympics on 20 August 2004.

 He broke the 44-year-old Olympics mark (by an Indian) held by Milkha Singh who set an Indian National Record with a timing of 45.73 s at the 1960 Rome
Olympics.

 He and his elder sister K. M. Beenamol made history when they became the first Indian siblings to win medals in a major international competition.

 They won medals at the Busan Asian Games (2002).

 While Binu won the men's 800 metres silver, his sister won the gold medal in the women's event.

 Binu received the Arjuna Award for the year 2006 for his achievements in the Indian athletics.
</item 4>

<item 5>
Evelyn Adiru
Evelyn Adiru (born 25 May 1964) is a former Ugandan middle-distance runner who specialised in 800 metres, and 1500 metres events.

 She won gold medal in 800 metres at the 1982 African Championships in Athletics in Cairo.

 She also competed for Uganda in the 1984 Summer Olympics in the same event, but did not progress to the finals.
</item 5>

<item 6>
Sanna Abubkheet
Sanna Abubkheet (Arabic: سناء أبو بخيت‎ ‎ ; born December 7, 1984 in Deir al-Balah) is a retired female Palestinian middle-distance runner, who specialized in
the 800 metres.

 Abubkheet qualified for the Palestinian squad in the women's 800 metres at the 2004 Summer Olympics in Athens by receiving a wild card entry slot from IAAF.

 Running against seven other athletes in heat three, Abubkheet crossed the finish line by more than half a minute behind leader and top medal favorite Kelly
Holmes of Great Britain with a seventh-place time in 2:32.10.

 Abubkheet failed to advance into the semifinals as she placed farther from two automatic slots for the next round and ranked no. 42 overall in the prelims.

 Building a historic milestone as the first female athlete from Gaza Strip to compete at the Olympics, Abubkheet was appointed by the Palestine Olympic
Committee to carry the nation's flag in the opening ceremony.
</item 6>

<item 7>
John Lindsay (Paralympian)
John Lindsay, OAM (born 29 January 1970) is an Australian Paralympic athlete from Melbourne.

 He competed in the 1988 Seoul games in distances ranging from 100 m to 800 m, but did not win any medals.

 At the 1992 Barcelona Games, he won a gold medal in the Men's 200 m TW3 event, for which he received a Medal of the Order of Australia, a silver medal in the
Men's 100 m TW3 event and a bronze medal in the Men's 400 m TW3 event.

 That year, he had a Victorian Institute of Sport scholarship.
</item 7>

<item 8>
Equatorial Guinea at the 2016 Summer Olympics
Equatorial Guinea competed at the 2016 Summer Olympics in Rio de Janeiro, Brazil, which was held from 5 to 21 August 2016.

 The country's participation at Rio marked its ninth appearance in the Summer Olympics since its début in the 1984 Summer Olympics.

 The delegation included two track and field athletes, Benjamin Enzema and Reïna-Flor Okori who both qualified for the Games through wildcard places since their
fastest times did not meet the required qualification standards, Okori was selected as the flag bearer for the opening and closing ceremonies.

 Enzema did not advance beyond the qualification round for the men's 800 metres event and Okori was unable to start the women's 100 metre hurdles contest.
</item 8>

<item 9>
Sinimole Paulose
Sinimol Paulose (born 24 June 1983) is an Indian middle distance runner.

 Her favorite events are 800 metres and 1500 metres.

 In 2009, she received Arjuna Award for her contribution to the Indian athletics.

 Sinimol was born in Piravom in Ernakulam district of Kerala state.

 She represented India at Beijing Olympics.
</item 9>

<item 10>
10,000 metres
The 10,000 metres or 10,000-meter run is a common long-distance track running event.

 The event is part of the athletics programme at the Olympic Games and the World Championships in Athletics and is common at championship level events.

 The race consists of 25 laps around an Olympic-sized track.

 It is less commonly held at track and field meetings, due to its duration.

 The 10,000 metres track race is usually distinguished from its road running counterpart, the 10K run, by its reference to the distance in metres rather than
kilometres.
</item 10>
=== END ITEM id=cb916c1df3ded8ad ===

=== ITEM id=ff73588afaf15717 ===
Gold sentences and facts:
Sentence 0:  The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.
  Fact 0.0: The pool lies directly south of Lichfield Cathedral.
  Fact 0.1: The pool has historically been important to the defence of the Cathedral Close.
Sentence 1:  The Diocese of Lichfield covers all of Staffordshire, much of Shropshire and part of the Black Country and West Midlands.
  Fact 1.0: The Diocese of Lichfield covers all of Staffordshire.
  Fact 1.1: The Diocese of Lichfield covers much of Shropshire.
  Fact 1.2: The Diocese of Lichfield covers part of the Black Country.
  Fact 1.3: The Diocese of Lichfield covers part of the West Midlands.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Lichfield Cathedral
Lichfield Cathedral is situated in Lichfield, Staffordshire, England.

 It is the only medieval English cathedral with three spires.

 The Diocese of Lichfield covers all of Staffordshire, much of Shropshire and part of the Black Country and West Midlands.

 The 99th and current Bishop of Lichfield is Michael Ipgrave who was appointed on 10 June 2016.
</item 2>

<item 3>
Black Country Rangers F.C.
Black Country Rangers Football Club is a football club based in Rowley Regis in the West Midlands.

 They are currently members of the West Midlands (Regional) League Premier Division and play at the Beeches in Cradley, groundsharing with Cradley Town.
</item 3>

<item 4>
Minster Pool
Minster Pool is a reservoir located between Bird Street and Dam Street in the heart of the city of Lichfield, Staffordshire in the United Kingdom.

 The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.

 The pool was originally formed in the 11th century when a boggy stream was dammed at its eastern end to drive a mill on Dam Street.

 The pool was used as a mill pond and fishery until 1856 when the mill was demolished; it has since been retained for public amenity.
</item 4>

<item 5>
Graph query row:
entity: Minster Pool
relationship: LOCATED_IN
connected_entity: Lichfield
connected_entity_labels: ['__KGBuilder__', '__Entity__', 'Place']
kind: city
country: None
held_on: None
regional_relationship: None
regional_connection: None
</item 5>

<item 6>
Black Country Living Museum
The Black Country Living Museum (formerly The Black Country Museum) is an open-air museum of rebuilt historic buildings in Dudley in the West Midlands of
England.

 It is located in the centre of the Black Country, 10 miles west of Birmingham.

 The museum occupies 105000 m2 of former industrial land partly reclaimed from a former railway goods yard, disused lime kilns, canal arm and former coal pits.
</item 6>

<item 7>
Walsall
Walsall ( ) is an industrial town in the West Midlands of England.

 It is located 8 miles north-west of the City of Birmingham and 6 miles east of the City of Wolverhampton.

 Historically a part of Staffordshire, Walsall is a component area of the West Midlands conurbation, and part of the Black Country.
</item 7>

<item 8>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 8>

<item 9>
Brierley Hill
Brierley Hill is a small town and electoral ward of the Dudley Metropolitan Borough, in the West Midlands of England, and is situated approximately 2.5 miles
south of central Dudley and 2 miles north of Stourbridge.

 Part of the Black Country, and in a heavily industrialised area of the Dudley Borough, it has a population of 13,935 at the 2011 census, and is best known for
glass and steel manufacturing, although the industry has declined considerably since the 1970s.

 One of the largest factories in the area was the Round Oak Steelworks, which was closed down and redeveloped in the 1980s to become the Merry Hill Shopping
Centre.

 Brierley Hill was originally in Staffordshire, but is now part of the West Midlands metropolitan county since its creation in 1974.
</item 9>

<item 10>
List of railway stations in the West Midlands
This is a list of railway stations within the West Midlands, a metropolitan county in central England which includes the cities of Birmingham, Coventry and
Wolverhampton.

 It includes all railway stations in the West Midlands that currently have regular timetabled train services, as well as certain stations outside the county
which are within the area supported by Transport for West Midlands (TfWM), formerly known as Centro.

 Transport within the West Midlands is subsidised by TfWM, who since 2006 have used the brand name Network West Midlands to demonstrate the 'joined-up' nature
of the regions bus and rail networks.
</item 10>
=== END ITEM id=ff73588afaf15717 ===

=== ITEM id=957e4428a6294449 ===
Gold sentences and facts:
Sentence 0: Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.
  Fact 0.0: Alice Malsenior Walker was born on February 9, 1944.
  Fact 0.1: Alice Malsenior Walker is an American novelist.
  Fact 0.2: Alice Malsenior Walker is an American short story writer.
  Fact 0.3: Alice Malsenior Walker is an American poet.
  Fact 0.4: Alice Malsenior Walker is an American activist.
Sentence 1: Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.
  Fact 1.0: Michael David Herr was born on April 13, 1940.
  Fact 1.1: Michael David Herr died on June 23, 2016.
  Fact 1.2: Michael David Herr was an American writer.
  Fact 1.3: Michael David Herr was an American war correspondent.
  Fact 1.4: Michael David Herr is known as the author of the memoir 'Dispatches', published in 1977.
  Fact 1.5: 'Dispatches' is a memoir of Michael David Herr's time as a correspondent for 'Esquire magazine' from 1967 to 1969.
  Fact 1.6: Michael David Herr was a correspondent during the Vietnam War.

Retrieved items:
<item 1>
Graph query row:
full_name: Michael David Herr
middle_name: David
</item 1>

<item 2>
Dick Herrick
The Very Rev Richard William Herrick (3 December 1913 - 5 May 1981) was an eminent Anglican priest in the 20th century.

 He was educated at King Edward VI School Retford and Leeds University and was initially a civil servant.

 He was ordained after a period of study at the College of the Resurrection, Mirfield in 1940.

 he held curacies at Duston and Portsea, Portsmouth before being appointed Vicar of St Michael’s, Northampton in 1947, a post he held for a decade.

 He was then a Canon Residentiary of Chelmsford Cathedral until 1978 when he was appointed Provost of Chelmsford.

 He died in post.
</item 2>

<item 3>
Graph query row:
full_name: Alice Malsenior Walker
middle_name: Malsenior
</item 3>

<item 4>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 4>

<item 5>
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
</item 5>

<item 6>
Alice R. Ballard
Alice R. Ballard.

 (born June 16, 1945, Florence, South Carolina) is an American ceramicist based in Greenville, South Carolina.

 Much of her work is characterized by the organic earthenware forms of closed containers, pinch pots, platters, pods, teapots, totems, small work, vessels and a
series she refers to as her white work.

 Much of her work is finished with terra sigillata.

 Alice considers her art to be “a reflection of [her] relationship with natural forms.

 It is often the metamorphosis of Nature's forms, as they change from season to season, that attracts her to that universal world in which differing life forms
share similar qualities."

 Her aesthetic heavily references the mother/child/germination metaphor and also explores the more evocative realm of wonder and awe.
</item 6>

<item 7>
Haniel Long
Haniel Clark Long (March 9, 1888 – October 17, 1956) was an American poet, novelist, publisher and academic.

 He is best known for his novella, "Interlinear to Cabeza de Vaca" (1936), a fictionalized account of the true story of a Spanish conquistador in 16th century
North America.
</item 7>

<item 8>
Tobe Levin
Alice Walker expressed appreciation for the text that shows worldwide solidarity with the novelist's literary abolition efforts in the early nineties.

 Levin has also teamed up with Maria Kiminta and photographer Britta Radike to publish a memoir and sourcebook, "Kiminta.

 A Maasai's Fight against Female Genital Mutilation."
</item 8>

<item 9>
Alice Walker
Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.

 She wrote the critically acclaimed novel "The Color Purple" (1982) for which she won the National Book Award and the Pulitzer Prize for Fiction.

 She also wrote the novels "Meridian" (1976) and "The Third Life of Grange Copeland" (1970), among other works.
</item 9>

<item 10>
Warrior Marks
Warrior Marks: Female Genital Mutilation and the Sexual Blinding of Women is a 1993 book by Alice Walker with Pratibha Parmar, who made an award-winning
documentary of the same name.

 Following on from her 1992 novel "Possessing the Secret of Joy", Walker undertakes a journey to parts of Africa where clitoridectomy is still practised.

 "Warrior Marks" is a harrowing work as Walker interviews women who have had the operation done and finally interviews a woman—circumcised herself—who performs
the operation.
</item 10>
=== END ITEM id=957e4428a6294449 ===

=== ITEM id=c9cc6bbac476b7ab ===
Gold sentences and facts:
Sentence 0:  The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and Viv Albertine, who went on to play in The Slits.
  Fact 0.0: The band never played live.
  Fact 0.1: The band never released any recordings.
  Fact 0.2: Like London SS, the band is more famed for the number of band members who later became well known.
  Fact 0.3: Like Masters of the Backside, the band is more famed for the number of band members who later became well known.
  Fact 0.4: Sid Vicious is listed as a band member who later became well known as a member of the Sex Pistols.
  Fact 0.5: Keith Levene is listed as a band member who later became well known.
  Fact 0.6: Keith Levene was an early member of The Clash.
  Fact 0.7: Keith Levene was later a member of Public Image Ltd.
  Fact 0.8: Palmolive is listed as a band member who later became well known.
  Fact 0.9: Viv Albertine is listed as a band member who later became well known.
  Fact 0.10: Viv Albertine went on to play in The Slits.
Sentence 1:  Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
  Fact 1.0: The band was interviewed by a fanzine named SKUM despite never playing live.
  Fact 1.1: In the SKUM interview, Sid Vicious proclaimed 'I'll just be the yob that I am now'.
Sentence 2: Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock band the Sex Pistols.
  Fact 2.0: Sid Vicious was born John Simon Ritchie.
  Fact 2.1: Sid Vicious was born on 10 May 1957.
  Fact 2.2: Sid Vicious died on 2 February 1979.
  Fact 2.3: Sid Vicious was an English bassist.
  Fact 2.4: Sid Vicious was an English vocalist.
  Fact 2.5: Sid Vicious was most famous as a member of the influential punk rock band the Sex Pistols.

Retrieved items:
<item 1>
Nancy Spungen
Nancy Laura Spungen (February 27, 1958 – October 12, 1978) was the American girlfriend of Sex Pistols bassist Sid Vicious and a figure of the 1970s punk rock
scene.

 Spungen's life and death have been the subject of controversy among music historians and fans of the Sex Pistols.
</item 1>

<item 2>
The Idols with Sid Vicious
The Idols with Sid Vicious is a concert album of former Sex Pistols bassist Sid Vicious performing with The Idols; though recorded in September 1978, it wasn't
released until 1993.

 Steve Dior provided a recording of this performance to the New Rose record label.

 It was originally released by the Fan Club division of New Rose.

 It was also released in Japan by Teichiku Records, under license from New Rose.
</item 2>

<item 3>
The Flowers of Romance (band)
The Flowers of Romance were an early punk band, formed in mid-1976 by Jo Faull and Sarah Hall.

 The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that
later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and
Viv Albertine, who went on to play in The Slits.

 Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
</item 3>

<item 4>
Sid Vicious
Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock
band the Sex Pistols.
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
Bodies (Sex Pistols song)
"Bodies" is a Sex Pistols song about the shock of abortion from the 1977 album "Never Mind the Bollocks, Here's the Sex Pistols".

 The lyrics contain very graphic imagery about a terminated fetus and feature a great deal of profanity for the time: the third and final verse begins with a
couplet in which the word "fuck" is repeated five times in rapid succession.

 Along with the later "Belsen Was a Gas," it is probably the most graphic and controversial Sex Pistols song in both its subject matter and style.

 Musically, it is also the fastest and heaviest song in the Sex Pistols canon — characterized by thudding drums, droning buzzsaw guitar, and shouted vocals.

 As such, it can be considered a significant antecedent to the genres of hardcore, thrash, and speed metal that was to emerge in the 1980s.
</item 6>

<item 7>
Revolution in the Classroom
"Revolution in the Classroom" was a single released by the Ex Pistols; despite its packaging, the record has nothing to do with the Sex Pistols.

 The single was released in 1989 on the STP Records label, pressed on various colours of vinyl.

 The song, along with its B-side "Schools Are Prisons", were both included on the Ex-Pistols album, "Deny".

 They also appear disguised as genuine Sex Pistols recordings on two Sex Pistols retrospective albums by former producer Dave Goodman, "Pirates of Destiny" (I
Swirled Records, 1989) and "We Have Cum For Your Children" (Skyclad Records, 1988).

Under the title "Revolution" the song appears on a German Sex Pistols compilation "Anarchy in the USA" from 1992 released on MBC records.

 CD 084-60322 LP 008-60321
</item 7>

<item 8>
Live at Winterland 1978 (Avengers album)
Live at Winterland 1978 is a live album by the Avengers.

 It was released through online music stores on February 16, 2010.

 The album features a recording of their set on January 14, 1978 at the Winterland Ballroom in San Francisco, CA, when they opened for the Sex Pistols on what
would become the Sex Pistols final show, before reuniting years later.

 The Sex Pistols' set has been released on an album of the same name.

 Songs from this album were previously released on a couple 7" bootlegs titled "Penelope" and "Summer of Hate".
</item 8>

<item 9>
Ex Pistols
The Ex Pistols were a scam act introduced in 1979 by one-time Sex Pistols producer Dave Goodman after his services were substituted for those of professional
producers.

 The Ex Pistols existed as a sound-alike band meant to mislead fans due to the similar music, name, and artwork, that led people to believe their songs were
actually new or lost Sex Pistols recordings.
</item 9>

<item 10>
Glen Matlock
Glen Matlock (born 27 August 1956) is an English musician best known for being the bass guitarist in the original line-up of the punk rock band the Sex Pistols.

 He is sometimes credited as a co-author on 10 of the 12 songs on "Never Mind the Bollocks, Here's the Sex Pistols", although he had left the band by the time
the record was released and the bass was played by Steve Jones.

 He also continues to make his own records and tour with various bands, including the Sex Pistols.
</item 10>
=== END ITEM id=c9cc6bbac476b7ab ===

=== ITEM id=471ce5b9a9737db7 ===
Gold sentences and facts:
Sentence 0:  The event was broadcast live on BBC One and was hosted by Graham Norton.
  Fact 0.0: The event was broadcast live on BBC One.
  Fact 0.1: Graham Norton hosted the event.
Sentence 1: Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.
  Fact 1.0: Graham William Walker was born on 4 April 1963.
  Fact 1.1: Graham William Walker is better known as Graham Norton.
  Fact 1.2: Graham Norton is an Irish television presenter.
  Fact 1.3: Graham Norton is an Irish radio presenter.
  Fact 1.4: Graham Norton is an Irish DJ.
  Fact 1.5: Graham Norton is an Irish comedian.
  Fact 1.6: Graham Norton is an Irish actor.
  Fact 1.7: Graham Norton is an Irish writer.

Retrieved items:
<item 1>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 1>

<item 2>
British Academy Children's Awards
The British Academy Children's Awards are presented in an annual award show hosted by the British Academy of Film and Television Arts (BAFTA).

 They have been awarded annually since 1996.

 Before that, children's awards were a part of the main British Academy Television Awards.
</item 2>

<item 3>
Graph query row:
person: Graham Norton
birth_date: 4 April 1963
</item 3>

<item 4>
Graham Norton
Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.

 He is a five-time BAFTA TV Award winner for his comedy chat show "The Graham Norton Show".

 Previously shown on BBC Two, it took the prestigious Friday night slot on BBC One from "The Jonathan Ross Show" in 2010.
</item 4>

<item 5>
British Academy Television Awards
The British Academy Television Awards, also known as the BAFTA Television Awards, are presented in an annual award show hosted by the British Academy of Film
and Television Arts (BAFTA).

 They have been awarded annually since 1955.

 It is the British equivalent to the Emmy Awards in the United States and Logie Awards in Australia.
</item 5>

<item 6>
2012 British Academy Television Awards
The 2012 British Academy Television Awards (formally known as the Arqiva British Academy Television Awards) were held on 27 May 2012 at the Royal Festival Hall
in London.
</item 6>

<item 7>
British Academy Television Awards 1997
The 1997 British Academy Television Awards were held on 29 April at the Royal Albert Hall in London, as a joint ceremony with the British Academy Film Awards.

 To date, it is the last occasion upon which the two sets awards have been given jointly at the same event; from 1998, the Film and Television Awards were given
at separate ceremonies, for the first time since 1968.
</item 7>

<item 8>
BAFTA Scotland
BAFTA in Scotland is the Scottish branch of the British Academy of Film and Television Arts.

 Formed in 1986, the branch holds two annual awards ceremonies recognising the achievement by performers and production staff in Scottish film, television and
video games.

 These Awards are separate from the British Academy Television Awards and British Academy Film Awards.
</item 8>

<item 9>
British Academy Television Award for Best Actress
This is a list of the British Academy Television Awards for Best Actress.

 The British Academy Television Awards began in 1955.

 The Best Actress award was initially given as an 'individual honour' without credit to a particular performance until 1969 when Wendy Craig won for her
performance in "Not in Front of the Children".

 Since 1970, nominees have been announced in addition to the winner and are listed with the winner highlighted in blue.

 The Actress category was split into Leading Actress and Supporting Actress starting in 2010.
</item 9>

<item 10>
British Academy Television Awards 1998
The 1998 British Academy Television Awards were held on 18 May at the Prince of Wales Theatre in London.

 The ceremony was broadcast on ITV, hosted by Bob Monkhouse, and it was the first occasion since 1968 that the Television Awards had been held separately from
the British Academy Film Awards, instead of as a joint ceremony.
</item 10>
=== END ITEM id=471ce5b9a9737db7 ===

=== ITEM id=c12bbb15b39deb36 ===
Gold sentences and facts:
Sentence 0: Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.
  Fact 0.0: Willi Forst was born Wilhelm Anton Frohs.
  Fact 0.1: Willi Forst's dates are 7 April 1903 to 11 August 1980.
  Fact 0.2: Willi Forst was an Austrian actor.
  Fact 0.3: Willi Forst was a screenwriter.
  Fact 0.4: Willi Forst was a film director.
  Fact 0.5: Willi Forst was a film producer.
  Fact 0.6: Willi Forst was a singer.
Sentence 1: Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.
  Fact 1.0: Elmer Clifton's dates are March 14, 1890 to October 15, 1949.
  Fact 1.1: Elmer Clifton was an American writer.
  Fact 1.2: Elmer Clifton was an American director.
  Fact 1.3: Elmer Clifton was an American actor.
  Fact 1.4: Elmer Clifton was active from the early silent days.

Retrieved items:
<item 1>
Winfred Overholser
Winfred Overholser (1892 – October 6, 1964) was an American psychiatrist, president of the American Psychiatric Association, and for 25 years the superintendent
of St. Elizabeths Hospital, a federal institution for the mentally ill in Washington, D.C.
</item 1>

<item 2>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 2>

<item 3>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 3>

<item 4>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 4>

<item 5>
Witmer Stone
Witmer Stone (September 22, 1866 – May 24, 1939) was an American ornithologist, botanist, and mammalogist, and was considered one of the last of the “great
naturalists.”

 Stone is remembered principally as an ornithologist.

 He was president of the American Ornithologists’ Union (AOU) 1920–23, and was editor of the AOU’s periodical "The Auk" 1912–1936.

 He spearheaded the production of the 4th edition of the AOU checklist, published in 1931.

 He worked for over 50 years in the Ornithology Department at the Academy of Natural Sciences of Philadelphia, eventually serving as Director of the
institution.

 Stone was one of the founding members of the Delaware Valley Ornithological Club (DVOC) in 1890 and was actively involved in the organization for the remainder
of his life.

 Stone was one of only two scientists (Joseph Grinnell was the other) to serve as president of both the AOU and the American Society of Mammalogists, and he
co-authored two popular books about mammals.
</item 5>

<item 6>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 6>

<item 7>
Alf Clausen
Alf Heiberg Clausen (born March 28, 1941) is an American film and television composer.

 He is best known for his work scoring many episodes of "The Simpsons", of which he had been the sole composer between 1990 and 2017.

 Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's
Day Off".
</item 7>

<item 8>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 8>

<item 9>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 9>

<item 10>
Kaiserjäger (film)
Kaiserjäger is a 1956 Austrian film directed by Willi Forst.
</item 10>
=== END ITEM id=c12bbb15b39deb36 ===

=== ITEM id=ecfcf7020225a3c0 ===
Gold sentences and facts:
Sentence 0: Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the west.
  Fact 0.0: Longkou was formerly known as Huang County.
  Fact 0.1: Longkou is a port city.
  Fact 0.2: Longkou is in northeastern Shandong province, China.
  Fact 0.3: Longkou faces the Bohai Sea to the north.
  Fact 0.4: Longkou faces the Laizhou Bay to the west.
Sentence 1:  The district faces East China Sea and is connected to Mawei district by the Min River.
  Fact 1.0: The district faces the East China Sea.
  Fact 1.1: The district is connected to Mawei district by the Min River.

Retrieved items:
<item 1>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 1>

<item 2>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 2>

<item 3>
Yandang Mountains
Yandang Mountains or Yandangshan (Chinese:  雁蕩山 ,  雁荡山 ,  "Yàndàng Shān",  "Wild Goose Pond Mountain(s)") refers, in the broad sense, to a coastal mountain
range in southeastern Zhejiang province in eastern China, covering much of the prefecture-level city of Wenzhou (from Pingyang County in the south to Yueqing
County in the northeast) and extending to the county-level city of Wenling in Taizhou prefecture.

 The mountain range is divided in two by the Oujiang River, the two parts being the North Yandang and South Yandang.

 More narrowly, Yandangshan is also used more narrowly to refer to Mount Yandang , a specific part of the North Yandang around an ancient caldera near a small
town of the same name (雁荡镇 , "Yàndàng Zhèn").

 The highest peaks of North Yandang are located here, and this is also the main tourist spot.
</item 3>

<item 4>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 4>

<item 5>
Dafeng District
Dafeng () is a coastal district under the administration of Yancheng, Jiangsu province, China.

 Located on the Jiangsu North Plain with a coastline of 112 km , Dafeng was historically one of the largest salt-making areas in China and now is famed for its
well preserved eco-system and numerous national conservation parks.

 The district has the largest national nature reserve for a rare deer species, Père David's Deer or Milu (麋鹿 ) in Chinese.

 It borders the prefecture-level city of Taizhou to the southwest.
</item 5>

<item 6>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 6>

<item 7>
Xujiacao Changle Station
Xujiacao Changle Station () is a station on Line 1 of the Ningbo Rail Transit that started operations on 30May 2014.

 It is situated over Wangchun Road () and Xinyuan Road () in the Haishu District of Ningbo City, Zhejiang Province, eastern China.
</item 7>

<item 8>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 8>

<item 9>
Changle District
(, Foochow Romanized: Diòng-lŏ̤h) is a district located in eastern Fujian province, China.

 Administered by Fuzhou, it occupies a land area of 648 km2 and a sea area of 1327 km2 .

 Changle was established in the sixth year of Emperor Wu-De (623 AD) during the Tang Dynasty, and it became a county-level city on February 18, 1994.

 The district faces East China Sea and is connected to Mawei district by the Min River.

 Due to an increase in businesses, the province is now one of the richest province in China.

 The city was upgraded into a district on August 2017 by a government proposal.
</item 9>

<item 10>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 10>
=== END ITEM id=ecfcf7020225a3c0 ===

=== ITEM id=d4f60b10f17e5076 ===
Gold sentences and facts:
Sentence 0:  Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
  Fact 0.0: Victoria died.
  Fact 0.1: The British throne passed to Victoria's eldest son Edward VII upon her death.
  Fact 0.2: Edward VII was a member of the House of Saxe-Coburg and Gotha through his father.
Sentence 1: The House of Saxe-Coburg and Gotha ( ; German: "") is a German dynasty that ruled the duchy of Saxe-Coburg and Gotha, which was one of the Ernestine duchies.
  Fact 1.0: The House of Saxe-Coburg and Gotha is a German dynasty.
  Fact 1.1: The House of Saxe-Coburg and Gotha ruled the duchy of Saxe-Coburg and Gotha.
  Fact 1.2: The duchy of Saxe-Coburg and Gotha was one of the Ernestine duchies.

Retrieved items:
<item 1>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 1>

<item 2>
House of Oldenburg
The House of Oldenburg is a European dynasty of North German origin.

 It is one of Europe's most influential royal houses, with branches that rule or have ruled in Denmark, Iceland, Greece, Norway, Russia, Sweden, Schleswig,
Holstein, and Oldenburg.

 The current Queen of Denmark and King of Norway, the former King of Greece, the consorts of Greece and the United Kingdom, as well as the first twelve names in
the line of succession to the British throne, are all patrilineal members of the Glücksburg branch of this house.
</item 2>

<item 3>
Principality of Calenberg
The Principality of Calenberg was a dynastic division of the Welf duchy of Brunswick-Lüneburg established in 1432.

 Calenberg was ruled by the House of Hanover from 1635 onwards; the princes received the ninth electoral dignity of the Holy Roman Empire in 1692.

 Their territory became the nucleus of the Electorate of Hanover, ruled in personal union with the Kingdom of Great Britain from 1714 onwards.

 The principality received its name from Calenberg Castle, a residence of the Brunswick dukes.
</item 3>

<item 4>
Succession to the British throne
Succession to the British throne is determined by descent, gender (for people born before October 2011), legitimacy, and religion.

 Under common law, the Crown is inherited by a sovereign's children or by a childless sovereign's nearest collateral line.

 The Bill of Rights 1689 and the Act of Settlement 1701, restrict succession to the throne to the legitimate Protestant descendants of Sophia of Hanover that
are in "communion with the Church of England".

 Spouses of Roman Catholics were disqualified from 1689 until the law was amended in 2015.

 Protestant descendants of those excluded for being Roman Catholics are eligible.
</item 4>

<item 5>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 5>

<item 6>
House of Hanover
The House of Hanover (or the Hanoverians ; German: "Haus Hannover" ) is a German royal dynasty that ruled the Electorate and then the Kingdom of Hanover, and
that also provided monarchs of Great Britain and Ireland from 1714 and ruled the United Kingdom until the death of Queen Victoria in 1901.

 Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
</item 6>

<item 7>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 7>

<item 8>
Duke of Bedford
Duke of Bedford (named after Bedford, England) is a title that has been created six times (for five distinct people) in the Peerage of England.

 The first and second creations came in 1414 in favour of Henry IV's third son, John, who later served as regent of France.

 He was made Earl of Kendal at the same time and was made Earl of Richmond later the same year.

 The titles became extinct on his death in 1435.

 The third creation came in 1470 in favour of George Neville, nephew of Warwick the Kingmaker.

 He was deprived of the title by Act of Parliament in 1478.

 The fourth creation came 1478 in favour of George, the third son of Edward IV.

 He died the following year at the age of two.

 The fifth creation came in 1485 in favour of Jasper Tudor, half-brother of Henry VI and uncle of Henry VII.

 He had already been created Earl of Pembroke in 1452.

 However, as he was a Lancastrian, his title was forfeited between 1461 and 1485 during the predominance of the House of York.
</item 8>

<item 9>
House of Saxe-Coburg and Gotha
The House of Saxe-Coburg and Gotha ( ; German: "") is a German dynasty that ruled the duchy of Saxe-Coburg and Gotha, which was one of the Ernestine duchies.

 It is a cadet branch of the Saxon House of Wettin.
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
=== END ITEM id=d4f60b10f17e5076 ===

=== ITEM id=d41735a9107bfba8 ===
Gold sentences and facts:
Sentence 0: "For the sedan variant of the car, see Volvo S40."
  Fact 0.0: The sedan variant of the car is described under Volvo S40.
Sentence 1: The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
  Fact 1.0: The Volvo S40 is a compact luxury car.
  Fact 1.1: The Volvo S40 is marketed by Volvo Cars.
  Fact 1.2: The Volvo S40 is produced by Volvo Cars.
  Fact 1.3: Volvo Cars is a Swedish manufacturer.
  Fact 1.4: The Volvo S40 was marketed and produced from 1995 to 2012.
  Fact 1.5: The Volvo S40 spanned two generations.

Retrieved items:
<item 1>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 1>

<item 2>
Volvo V50
"For the sedan variant of the car, see Volvo S40."
</item 2>

<item 3>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 3>

<item 4>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 4>

<item 5>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 5>

<item 6>
Volvo Car Gent
Volvo Car Gent (until 2007 known as Volvo Cars Europe Industry N.V.) is an automobile manufacturer located in the port district of Ghent in western Belgium.

 In recent years, the Ghent plant has become the largest Volvo plant in terms of the number of cars produced.

 The company is a wholly owned subsidiary of the Swedish automaker, Volvo Personvagnar AB (Volvo Cars).

 Legally it is part of Volvo Car Belgium NV.

 Its activities comprise welding, painting and final assembly.

 The plant became the only car manufacturing plant in Flanders after Ford Genk closed at the end of 2014.
</item 6>

<item 7>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 7>

<item 8>
Volvo 300 Series
The Volvo 300 Series is a rear wheel drive small family car sold as both a hatchback and (later) a conventional saloon from 1976 to 1991.

 It was launched in the Netherlands shortly after Volvo acquired a major stake in the passenger car division of DAF in 1973.

 The series consisted of the Volvo 340 (previously 343/345) and the later Volvo 360.
</item 8>

<item 9>
Volvo C30
The Volvo C30 is a near-luxury compact hatchback, manufactured and marketed by Volvo Cars for model years 2006-2013, available across its single generation as a
three-door hatchback.

 Powered by inline-four and straight-five engines, the C30 is variant of the Volvo S40/V50/C70 range, sharing the same Ford C1/Volvo P1 platform.

 Volvo marketed the C30 as a premium hatchback / sports coupe.
</item 9>

<item 10>
Volvo 440/460
The Volvo 440 and 460 are versions of a small family car produced by the Swedish manufacturer Volvo between 1987 and 1996.

 The 440 was introduced in 1987, whilst the 460 followed in 1989.

 They were built at the NedCar factory in Born, The Netherlands.

 The 440 was a five door front wheel drive hatchback and the 460 a saloon.

 They shared many components with the already successful Volvo 480, including floorpan, front and rear suspension, engines, transmission and braking systems.
</item 10>
=== END ITEM id=d41735a9107bfba8 ===

=== ITEM id=ce8bd8f903311d70 ===
Gold sentences and facts:
Sentence 0: Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.
  Fact 0.0: Dangal's English title is "Wrestling competition".
  Fact 0.1: Dangal is a 2016 film.
  Fact 0.2: Dangal is an Indian film.
  Fact 0.3: Dangal is a Hindi-language film.
  Fact 0.4: Dangal is a biographical sports drama film.
  Fact 0.5: Dangal is loosely based on the story of Mahavir Singh Phogat.
  Fact 0.6: Mahavir Singh Phogat is an amateur wrestler.
  Fact 0.7: Mahavir Singh Phogat trains his daughters Geeta Phogat and Babita Kumari.
  Fact 0.8: Mahavir Singh Phogat trains his daughters to be world-class wrestlers.
Sentence 1: The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
  Fact 1.0: The Man from Snowy River II is a 1988 film.
  Fact 1.1: The Man from Snowy River II is an Australian drama film.
  Fact 1.2: The Man from Snowy River II is the sequel to the 1982 film "The Man from Snowy River".

Retrieved items:
<item 1>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 1>

<item 2>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 2>

<item 3>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 3>

<item 4>
The Man from Snowy River (TV series)
The Man from Snowy River is an Australian adventure drama television series based on Banjo Paterson's poem "The Man from Snowy River".

 Released in Australia as "Banjo Paterson's The Man from Snowy River", the series was subsequently released in both the United States and the United Kingdom as
"Snowy River: The McGregor Saga".
</item 4>

<item 5>
The Man from Snowy River (soundtrack)
"The Man from Snowy River is the original motion picture soundtrack from the 1982 film "The Man from Snowy River".
</item 5>

<item 6>
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
</item 6>

<item 7>
The Man from Snowy River II
The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
</item 7>

<item 8>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 8>

<item 9>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 9>

<item 10>
The Man from Snowy River (poem)
"The Man from Snowy River" is a poem by Australian bush poet Banjo Paterson.

 It was first published in "The Bulletin", an Australian news magazine, on 26 April 1890, and was published by Angus & Robertson in October 1895, with other
poems by Paterson, in "The Man from Snowy River and Other Verses".
</item 10>
=== END ITEM id=ce8bd8f903311d70 ===

=== ITEM id=b421e7862478a9a8 ===
Gold sentences and facts:
Sentence 0: Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
  Fact 0.0: Grantley Jonathan "Grant" Nicholas was born on 12 November 1967.
  Fact 0.1: Grant Nicholas is a Welsh musician.
  Fact 0.2: Grant Nicholas is best known as the lead singer of the rock band Feeder.
  Fact 0.3: Grant Nicholas is best known as the lead guitarist of the rock band Feeder.
Sentence 1: Danny Shirley (born August 12, 1956) is an American country music singer.
  Fact 1.0: Danny Shirley was born on August 12, 1956.
  Fact 1.1: Danny Shirley is an American country music singer.

Retrieved items:
<item 1>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 1>

<item 2>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 2>

<item 3>
Yorktown Heights (album)
Yorktown Heights was the only solo album by Feeder frontman Grant Nicholas.

 The album was released on 11 August 2014 in the United Kingdom and recorded during 2013, being a year after Nicholas announced at Feeder's final show of 2012
at the Brixton Academy that the band would be taking a break.

 It was with the first airplay of ""Soul Mates"" on the BBC Radio 2 show "The Dermot O'Leary Show" on 31 May 2014, that Grant's solo career made its radio
debut.
</item 3>

<item 4>
Don't Stop Believing (TV series)
Don't Stop Believing is a British television talent show that aired on Channel 5 in summer 2010.

 It was inspired by the musical comedy-drama "Glee", which airs in the United States on the Fox network.

 The series featured live shows in which musical performance groups competed against each other, with viewers voting on the winner.

 Solo singers were also sought to join a group to represent the United Kingdom on the American glee club circuit.

 The show was hosted by Emma Bunton, and judged by Anastacia, Duncan James, Tamzin Outhwaite and Charles "Chucky" Klapow.

 The programme was shown in simulcast on Irish TV channel 3e and repeated a week later on parent channel TV3 Ireland.

 The show was not renewed for a second series due to low ratings.
</item 4>

<item 5>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 5>

<item 6>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 6>

<item 7>
Fayard Nicholas
Fayard Antonio Nicholas (October 20, 1914 – January 24, 2006) was an American choreographer, dancer and actor.

 He and his younger brother Harold Nicholas made up the Nicholas Brothers tap-dance duo, who starred in the MGM musicals "An All-Colored Vaudeville Show"
(1935), "Stormy Weather" (1943), "The Pirate" (1948), and "Hard Four" (2007).

 The Nicholas brothers also starred in the 20th Century-Fox musicals "Down Argentine Way" (1940), "Sun Valley Serenade" (1941), and "Orchestra Wives" (1942).
</item 7>

<item 8>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 8>

<item 9>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 9>

<item 10>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 10>
=== END ITEM id=b421e7862478a9a8 ===
