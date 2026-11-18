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

=== ITEM id=7409bc3e8f7c36e0 ===
Gold sentences and facts:
Sentence 0:  Her personal best throw is 58.87 metres, achieved at the 2002 Asian Games in Busan.
  Fact 0.0: Her personal best throw is 58.87 metres.
  Fact 0.1: She achieved her personal best throw at the 2002 Asian Games.
  Fact 0.2: The 2002 Asian Games were held in Busan.
Sentence 1:  A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.
  Fact 1.0: A total of 460 athletes took part in the competition.
  Fact 1.1: The athletes comprised 281 men and 179 women.
  Fact 1.2: The athletes came from 39 nations.

Retrieved items:
<item 1>
Pavel Lyzhyn
Pavel Kanstantsinavich Lyzhyn (Belarusian: Павел Канстанцінавіч Лыжын, Pavieł Łyžyn , born 24 March 1981 in Vysokaye) is a Belarusian shot putter.

 His personal best throw is 21.21 metres from 2010.

 He threw a personal best throw of 20.98 metres at the 2008 Olympics in Beijing which originally translated into a fourth place, just 5 cm short of the bronze
medal winner and 11 cm short of the silver.

 However, on 25 November 2016 the IOC disqualified him from the 2008 Olympic Games and struck his results from the record for failing a drugs test in a
re-analysis of his doping sample from 2008.
</item 1>

<item 2>
Luan Zhili
Luan Zhili (born 6 January 1973) is a female discus thrower from PR China.

 She finished ninth at the 1997 World Championships and won the 1998 Asian Games, the latter in a personal best throw of 63.43 metres.

 The Chinese, and Asian, record is currently held by Xiao Yanling with 71.68 metres.
</item 2>

<item 3>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 3>

<item 4>
Daniel Ståhl
In June 2017, Ståhl managed a throw of 71.29 metres in Sollentuna, setting a new personal best and improving the Swedish record set in 1984 by Rickard Bruch by
three centimetres.

 Ståhl's throw was the farthest discus throw in the world for four years.
</item 4>

<item 5>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 5>

<item 6>
John Lindsay (Paralympian)
At the 2004 Athens Games, he came seventh in the first round of the Men's 100 m T53 event and sixth in the third round of the Men's 200 m T53 event.

 He was an Australian Institute of Sport scholarship holder in 1995 and 2000.
</item 6>

<item 7>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 7>

<item 8>
Lee Young-sun
Lee Young-Sun (born 21 February 1974) is a retired female javelin thrower from South Korea.

 Her personal best throw is 58.87 metres, achieved at the 2002 Asian Games in Busan.
</item 8>

<item 9>
Masayoshi Yamamoto
Masayoshi Yamamoto (Japanese: 山本 雅賢) is a Japanese male artistic gymnast, representing his nation at international competitions.

 He won the gold medal in the Pommel horse event at the 2014 Asian Games in Incheon, South Korea.
</item 9>

<item 10>
Zhao Qinggang
Zhao Qinggang (; born 24 July 1985) is a Chinese track and field athlete who competes in the javelin throw.

 His 89.15 m personal best throw, set in 2014, is the Chinese record.

 It was also the Asian record until surpassed in 2017.

 He represented his country at the 2013 World Championships in Athletics.

 He is the 2014 Asian Games champion.

 At the East Asian Games he was the runner-up in 2009 and winner in 2013.

 He is a two-time Chinese champion (2012 and 2013) and was the winner of the 2013 National Games of China.
</item 10>
=== END ITEM id=7409bc3e8f7c36e0 ===

=== ITEM id=5c7c546635b11e9c ===
Gold sentences and facts:
Sentence 0: Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer, musician, director, actor, and comedian who holds dual British and American citizenship.
  Fact 0.0: Christopher Haden-Guest is the 5th Baron Haden-Guest.
  Fact 0.1: Christopher Haden-Guest was born on February 5, 1948.
  Fact 0.2: Christopher Haden-Guest is simply known as Christopher Guest.
  Fact 0.3: Christopher Guest is an English-American screenwriter.
  Fact 0.4: Christopher Guest is an English-American composer.
  Fact 0.5: Christopher Guest is an English-American musician.
  Fact 0.6: Christopher Guest is an English-American director.
  Fact 0.7: Christopher Guest is an English-American actor.
  Fact 0.8: Christopher Guest is an English-American comedian.
  Fact 0.9: Christopher Guest holds dual British and American citizenship.
Sentence 1:  Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.0: Christopher Guest is most widely known in Hollywood for having written his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.1: Christopher Guest is most widely known in Hollywood for having directed his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.2: Christopher Guest is most widely known in Hollywood for having starred in his series of comedy films shot in mock-documentary (mockumentary) style.
Sentence 2:  He also directed dark comedies, children's movies, and historical pictures.
  Fact 2.0: Christopher Guest directed dark comedies.
  Fact 2.1: Christopher Guest directed children's movies.
  Fact 2.2: Christopher Guest directed historical pictures.

Retrieved items:
<item 1>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 1>

<item 2>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 2>

<item 3>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 3>

<item 4>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 4>

<item 5>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 5>

<item 6>
David Jablin
David Jablin is an award winning comedy producer and director and one of the pioneers of original television programming for Pay Television.

 In 1981 he created and produced the innovative comedy anthology series "Likely Stories" for HBO/Cinemax, which featured the directing debuts of Rob Reiner,
Danny DeVito, Christopher Guest, and Billy Crystal.
</item 6>

<item 7>
Vegas Vacation
Vegas Vacation is a 1997 American comedy film directed by Stephen Kessler.

 It is the fourth installment in "National Lampoon"’s "Vacation" film series, and was written by Elisa Bell, based on a story by Bell and Bob Ducsay.

 The film stars Chevy Chase, Beverly D'Angelo and Randy Quaid, with Ethan Embry and Marisol Nichols as Griswold children Rusty and Audrey.

 The film opened at #4 at the box office and grossed over $36.4 million domestically.

 "Vegas Vacation" became the first theatrical "Vacation" film not to carry the National Lampoon label or a screenwriting credit from John Hughes.
</item 7>

<item 8>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 8>

<item 9>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 9>

<item 10>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 10>
=== END ITEM id=5c7c546635b11e9c ===

=== ITEM id=1b3ce6127486464b ===
Gold sentences and facts:
Sentence 0:  Her only novel, "The Bone People", won the Booker Prize in 1985.
  Fact 0.0: Her only novel is 'The Bone People'.
  Fact 0.1: 'The Bone People' won the Booker Prize.
  Fact 0.2: 'The Bone People' won the Booker Prize in 1985.
Sentence 1:  His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science fiction's top U.S. prizes in consecutive years.
  Fact 1.0: Card's novel 'Ender's Game' was published in 1985.
  Fact 1.1: 'Speaker for the Dead' is the sequel to 'Ender's Game'.
  Fact 1.2: 'Speaker for the Dead' was published in 1986.
  Fact 1.3: 'Ender's Game' won the Hugo Award.
  Fact 1.4: 'Ender's Game' won the Nebula Award.
  Fact 1.5: 'Speaker for the Dead' won the Hugo Award.
  Fact 1.6: 'Speaker for the Dead' won the Nebula Award.
  Fact 1.7: Card is the only author to win both science fiction's top U.S. prizes in consecutive years.

Retrieved items:
<item 1>
Ender in Exile
Ender in Exile is a science fiction novel by Orson Scott Card, part of the "Ender's Game" series, published on November 11, 2008.

 It takes place between the two award-winning novels: "Ender's Game" and "Speaker for the Dead".

 It could also be considered a parallel novel to the first three sequels in the Shadow Saga, since the entirety of this "trilogy" takes place in the span of
"Ender in Exile".

 The novel concludes a dangling story line of the Shadow Saga, while it makes several references to events that take place during the Shadow Saga.

 From yet another perspective, the novel expands (or "replaces") the last chapter of the original novel "Ender's Game".

 On the one hand, it fills the gap right before the last chapter, and on the other hand, it fills the gap between the last chapter and the original (first)
sequel (both named "Speaker for the Dead").
</item 1>

<item 2>
Ender Wiggin
Andrew "Ender" Wiggin is a fictional character from Orson Scott Card's 1985 science fiction novel "Ender's Game" and its sequels ("Speaker for the Dead",
"Xenocide", "Children of the Mind", "Ender in Exile"), as well as in the first part of the spin-off series, "Ender's Shadow".

 The book series itself is an expansion, with some changes to detail, of Card's 1977 short story "Ender's Game."
</item 2>

<item 3>
Orson Scott Card bibliography
This is a list of the works of Orson Scott Card.

 This list does not include criticisms, reviews, or related material written by Card.

 Orson Scott Card is the author of The Ender saga and Homecoming Saga among many other works.
</item 3>

<item 4>
Ender in Exile
"Ender in Exile" begins one year after Ender has won the bugger war, and begins with the short story "Ender's Homecoming" from Card's webzine Intergalactic
Medicine Show.

 Other short stories that were published elsewhere are included as chapters of the novel.
</item 4>

<item 5>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 5>

<item 6>
A War of Gifts: An Ender Story
A War of Gifts: An Ender Story (2007) is a science fiction novel by Orson Scott Card.

 This book is set in Card's Ender's Game series and takes place during Ender Wiggin's time at Battle School as described in Card's novels "Ender's Game" and
"Ender's Shadow".
</item 6>

<item 7>
Shadow of the Hegemon
Shadow of the Hegemon (2001) is the second novel in the "Ender's Shadow" series (often called the Bean Quartet) by Orson Scott Card.

 It is also the sixth novel in the "Ender's Game" series.

 It is told mostly from the point of view of Bean, a largely peripheral character in the original novel "Ender's Game" but the central protagonist of the
parallel narrative "Ender's Shadow."

 "Shadow of the Hegemon" was nominated for a Locus Award in 2002.
</item 7>

<item 8>
Ender's Game (short story)
"Ender's Game" is a story by Orson Scott Card.

 It first appeared in the August 1977 issue of "Analog" magazine and was later expanded into the novel "Ender's Game".

 Although the foundation of the Ender's Game series, the short story is not properly part of the "Ender's Game" universe, as there are many discrepancies in
continuity.
</item 8>

<item 9>
The Bone People
The Bone People (styled by the writer and in some editions as the bone people) is a Booker Prize-winning 1984 novel by New Zealand writer Keri Hulme.
</item 9>

<item 10>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 10>
=== END ITEM id=1b3ce6127486464b ===

=== ITEM id=068e036792b3c47f ===
Gold sentences and facts:
Sentence 0: The Doris Bither case, also known as the Entity hauntings, was an alleged haunting which occurred in 1974 at Culver City, California where a woman named Doris Bither alleged the ghosts of three Asian men were raping her.
  Fact 0.0: The Doris Bither case is also known as the Entity hauntings.
  Fact 0.1: The Doris Bither case was an alleged haunting.
  Fact 0.2: The alleged haunting occurred in 1974.
  Fact 0.3: The alleged haunting occurred at Culver City, California.
  Fact 0.4: Doris Bither alleged that the ghosts of three Asian men were raping her.
Sentence 1: The Entity is a 1982 American horror film directed by Sidney J. Furie and written by Frank De Felitta, who adapted his 1978 novel of the same name.
  Fact 1.0: The Entity is a 1982 American horror film.
  Fact 1.1: The Entity was directed by Sidney J. Furie.
  Fact 1.2: The Entity was written by Frank De Felitta.
  Fact 1.3: Frank De Felitta adapted his 1978 novel of the same name for The Entity.
Sentence 2:  Like the novel, the film is based on the 1974 Doris Bither case.
  Fact 2.0: The film is based on the 1974 Doris Bither case.
  Fact 2.1: The novel is also based on the 1974 Doris Bither case.

Retrieved items:
<item 1>
Midnight (1934 film)
Midnight is a 1934 American drama film directed by Chester Erskine and starring Sidney Fox, O.P. Heggie, Henry Hull and Margaret Wycherly.

 The film was produced for Universal and was shot on a modest budget of $50,000 at Thomas Edison Studios, which producer/director Chester Erskine had re-opened
specifically for the shoot.
</item 1>

<item 2>
The Snake Woman
The Snake Woman is a low budget 1961 British horror film directed by Sidney J. Furie and starring Susan Travers, John McCarthy.
</item 2>

<item 3>
Obsession (1976 film)
Obsession is a 1976 psychological thriller/mystery film directed by Brian De Palma, starring Cliff Robertson, Geneviève Bujold, John Lithgow, and Stocker
Fontelieu.

 The screenplay was by Paul Schrader, from a story by De Palma and Schrader.

 Bernard Herrmann provided the film's soundtrack prior to his death in 1975.

 The story is about a New Orleans businessman who is haunted by guilt following the death of his wife and daughter during a kidnapping-rescue attempt.

 Years after the tragedy, he meets and falls in love with a young woman who is the exact look-alike of his long dead wife.
</item 3>

<item 4>
Doris Bither case
The Doris Bither case, also known as the Entity hauntings, was an alleged haunting which occurred in 1974 at Culver City, California where a woman named Doris
Bither alleged the ghosts of three Asian men were raping her.

 The case inspired Frank De Felitta's 1978 book "The Entity" which was made into a 1982 film of the same name starring Barbara Hershey.
</item 4>

<item 5>
Little Fauss and Big Halsy
Little Fauss and Big Halsy is a 1970 film directed by Sidney J. Furie, starring Robert Redford and Michael J. Pollard, also featuring Lauren Hutton, Noah Beery,
Jr. and Lucille Benson.
</item 5>

<item 6>
Doctor Blood's Coffin
Doctor Blood's Coffin is a 1961 British horror film directed by Sidney J. Furie.

 Kieron Moore stars as a mad scientist who reanimates the dead.
</item 6>

<item 7>
The Entity
The Entity is a 1982 American horror film directed by Sidney J. Furie and written by Frank De Felitta, who adapted his 1978 novel of the same name.

 It stars Barbara Hershey as a woman who is raped and tormented by an invisible assailant.

 Despite being filmed and planned for a release in 1981, the movie was not released in worldwide theaters until September 1982 and February 1983 in the United
States.

 Like the novel, the film is based on the 1974 Doris Bither case.
</item 7>

<item 8>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 8>

<item 9>
Fatal Vision (miniseries)
Fatal Vision is a 1984 American television miniseries based on the account, in the book of the same name, of the murders in 1970 at Fort Bragg of the wife and
daughters of U.S. Army officer Jeffrey R. MacDonald.
</item 9>

<item 10>
The Rocky Horror Picture Show
The Rocky Horror Picture Show is a 1975 British-American musical screwball comedy horror film by 20th Century Fox produced by Lou Adler and Michael White and
directed by Jim Sharman.

 The screenplay was written by Sharman and actor Richard O'Brien, who appears in the film, which is based on the 1973 musical stage production "of the same
title", with music, book, and lyrics by O'Brien.

 The production is a parody tribute to the science fiction and horror B movies of the 1930s through to the early 1970s.

 Along with O'Brien, the film stars Tim Curry, Susan Sarandon, and Barry Bostwick and is narrated by Charles Gray with cast members from the original Royal
Court Theatre, Roxy Theatre, and Belasco Theatre productions.
</item 10>
=== END ITEM id=068e036792b3c47f ===

=== ITEM id=30bf71e09934f036 ===
Gold sentences and facts:
Sentence 0: Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.
  Fact 0.0: Semyon Aronovich Gershgorin was born on August 24, 1901.
  Fact 0.1: Semyon Aronovich Gershgorin died on May 30, 1933.
  Fact 0.2: Semyon Aronovich Gershgorin was Soviet.
  Fact 0.3: Semyon Aronovich Gershgorin was born in Pruzhany, Belarus, Russian Empire.
  Fact 0.4: Semyon Aronovich Gershgorin was a mathematician.
Sentence 1:  He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after him.
  Fact 1.0: Semyon Aronovich Gershgorin was a mathematician.
  Fact 1.1: Semyon Aronovich Gershgorin was noted for his work on the decidability of various algebraic groups.
  Fact 1.2: Malcev algebras are generalisations of Lie algebras.
  Fact 1.3: Malcev algebras are named after Semyon Aronovich Gershgorin.

Retrieved items:
<item 1>
Alexander Artemev
Alexander Vladimirovich "Sasha" Artemev (Russian: Александр Владимирович "Саша" Артемьев , Belarusian: Аляксандр Уладзіміравіч Арцем'еў ; born August 29, 1985)
is a retired American artistic gymnast.

 Artemev was a member of the bronze medal-winning U.S. team at the 2008 Olympic Games.

 He is the 2006 all-around U.S. national champion.

 Known for his ability on pommel horse, he is the 2007 and 2008 U.S. national champion on the pommel horse and won the bronze medal on the event at the 2006
World Championships.
</item 1>

<item 2>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 2>

<item 3>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 3>

<item 4>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 4>

<item 5>
Graph query row:
person1: Semyon Aranovich Gershgorin
field1: ['mathematician', 'student', 'Professor']
person2: Anatoly Maltsev
field2: mathematician
</item 5>

<item 6>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 6>

<item 7>
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
</item 7>

<item 8>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 8>

<item 9>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 9>

<item 10>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 10>
=== END ITEM id=30bf71e09934f036 ===

=== ITEM id=8efd7b5b579485db ===
Gold sentences and facts:
Sentence 0: Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.
  Fact 0.0: Return to Never Land is also known as Peter Pan 2.
  Fact 0.1: Return to Never Land is also known as Peter Pan In: Return to Never Land.
  Fact 0.2: Return to Never Land is a 2002 American film.
  Fact 0.3: Return to Never Land is an animated musical fantasy-adventure film.
  Fact 0.4: Return to Never Land was produced by Walt Disney Television Animation.
  Fact 0.5: Return to Never Land was released by Walt Disney Pictures.
  Fact 0.6: Return to Never Land was distributed by Walt Disney Studios Motion Pictures.
Sentence 1:  The film was co-produced by Walt Disney Pictures and Mayhem Pictures.
  Fact 1.0: Return to Never Land was co-produced by Walt Disney Pictures.
  Fact 1.1: Return to Never Land was co-produced by Mayhem Pictures.

Retrieved items:
<item 1>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 1>

<item 2>
Josann McGibbon
Josann McGibbon is an American screenwriter working in partnership with Sara Parriott.

 The team's first major success as a screenwriter was the early Brad Pitt film, "The Favor".

 Their biggest hits since then include "Three Men and a Little Lady" and "Runaway Bride".

 In 2007, McGibbon and Parriott co-wrote and produced the hit Debra Messing miniseries, "The Starter Wife".

 "The Starter Wife" received 10 Emmy nominations in 2007, including for best screenwriting, and won one Emmy Award.

 It was also nominated for Golden Globe and Writers Guild awards, and was then produced as a series, also on USA Network.

 McGibbon and Parriott wrote and co-produced the Disney Channel movie, "Descendants" which was directed by Kenny Ortega and premiered in July, 2015.

 On February, 2013, it won the Writers Guild of America Award in television for Outstanding Children's Long Form.

 They then wrote and executive-produced "Descendants 2", also directed by Kenny Ortega, which was simulcast on the Disney Channel, ABC, and the other cable
channels owned by Disney-ABC in July, 2017.
</item 2>

<item 3>
Return to Never Land
Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by
Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.

 The film is a sequel to Walt Disney Feature Animation 1953 film "Peter Pan", It is based on J. M. Barrie's novel "Peter and Wendy", and had a worldwide gross
of $109 million.
</item 3>

<item 4>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 4>

<item 5>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 5>

<item 6>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 6>

<item 7>
Peter Pan: Adventures in Never Land
Peter Pan: Adventures in Never Land (also known as Peter Pan in Disney's Return to Never Land) is a 2002 video game based on the Peter Pan Disney franchise.
</item 7>

<item 8>
Us (film)
Us is a 1991 television movie broadcast on CBS, produced, written and directed by Michael Landon.

 Landon also starred in the film, along with Barney Martin and Casey Peterson.

 It was a pilot for what would have been Landon's fourth consecutive television series; Landon's death that year precluded its going ahead.

 Landon played Jeff Hayes, a man just released from prison after serving many years due to being wrongfully convicted of killing a wealthy man's wife.
</item 8>

<item 9>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 9>

<item 10>
Never Land Books
The Never Land Books or Never Land Adventures are a series of short chapter books set in Never Land, the home of Peter Pan.

 They are based on the situations and characters established in the novel "Peter and the Starcatchers" and its sequels.

 Like the novels, they are written by Pulitzer Prize-winning humorist Dave Barry and suspense novelist Ridley Pearson, and illustrated by Greg Call.

 Although five books were planned, only three were published, in 2006–2008.

 The stories focus on supporting characters from the novels, such as the Indians, mermaids, pirates, and Lost Boys.

 They include:
</item 10>
=== END ITEM id=8efd7b5b579485db ===

=== ITEM id=38016e1ec0bc3a96 ===
Gold sentences and facts:
Sentence 0:  It covers an area of 1,383 sqkm south east of Melbourne, stretching from outer suburban Pakenham through rural areas to the coast at Inverloch and Phillip Island.
  Fact 0.0: The area covers 1,383 sqkm.
  Fact 0.1: The area is south east of Melbourne.
  Fact 0.2: The area stretches from outer suburban Pakenham through rural areas.
  Fact 0.3: The area stretches to the coast at Inverloch and Phillip Island.
Sentence 1:  It includes the towns of Bass, Cape Patterson, Corinella, Cowes, Grantville, Inverloch, Kilcunda, Koo Wee Rup, Lang Lang, Rhyll, San Remo, Ventnor, Wonthaggi, and part of suburban Pakenham.
  Fact 1.0: The area includes the town of Bass.
  Fact 1.1: The area includes the town of Cape Patterson.
  Fact 1.2: The area includes the town of Corinella.
  Fact 1.3: The area includes the town of Cowes.
  Fact 1.4: The area includes the town of Grantville.
  Fact 1.5: The area includes the town of Inverloch.
  Fact 1.6: The area includes the town of Kilcunda.
  Fact 1.7: The area includes the town of Koo Wee Rup.
  Fact 1.8: The area includes the town of Lang Lang.
  Fact 1.9: The area includes the town of Rhyll.
  Fact 1.10: The area includes the town of San Remo.
  Fact 1.11: The area includes the town of Ventnor.
  Fact 1.12: The area includes the town of Wonthaggi.
  Fact 1.13: The area includes part of suburban Pakenham.
Sentence 2: Pakenham is a satellite suburb of Melbourne on the edge of the West Gippsland region of Victoria, Australia, 56 km south-east of Melbourne's Central Business District, located within the Shire of Cardinia local government area.
  Fact 2.0: Pakenham is a satellite suburb of Melbourne.
  Fact 2.1: Pakenham is on the edge of the West Gippsland region of Victoria, Australia.
  Fact 2.2: Pakenham is 56 km south-east of Melbourne's Central Business District.
  Fact 2.3: Pakenham is located within the Shire of Cardinia local government area.

Retrieved items:
<item 1>
Electoral district of Emerald Hill
Emerald Hill was an electoral district of the Victorian Legislative Assembly, Australia.

 It covered part of the inner-city suburb South Melbourne and consisted part of the previous Electoral district of South Melbourne which was abolished in 1859.

 (The other part of the South Melbourne electorate became the Electoral district of Sandridge).
</item 1>

<item 2>
Electoral district of Bass
The Electoral district of Bass is an electoral district of the Victorian Legislative Assembly.

 It covers an area of 1,383 sqkm south east of Melbourne, stretching from outer suburban Pakenham through rural areas to the coast at Inverloch and Phillip
Island.

 It includes the towns of Bass, Cape Patterson, Corinella, Cowes, Grantville, Inverloch, Kilcunda, Koo Wee Rup, Lang Lang, Rhyll, San Remo, Ventnor, Wonthaggi,
and part of suburban Pakenham.

 It lies within the Eastern Victoria Region of the upper house, the Legislative Council.
</item 2>

<item 3>
Pakenham, Victoria
Pakenham is a satellite suburb of Melbourne on the edge of the West Gippsland region of Victoria, Australia, 56 km south-east of Melbourne's Central Business
District, located within the Shire of Cardinia local government area.

 Pakenham recorded a population of 46,421 at the 2016 Census.
</item 3>

<item 4>
Electoral district of Braidwood
Braidwood was an electoral district of the Legislative Assembly in the Australian state of New South Wales between 1859 and 1904, which included the town of
Braidwood.

 It replaced parts of the electoral district of United Counties of Murray and St Vincent and the electoral district of Southern Boroughs.

 In 1904 it was largely absorbed into the electoral district of Queanbeyan.
</item 4>

<item 5>
Beaconsfield, Victoria
Beaconsfield is a satellite suburb of Melbourne, Victoria, Australia, 46 km south-east of Melbourne's Central Business District, located within the Shire of
Cardinia local government area.

 Beaconsfield recorded a population of 6,714 at the 2016 Census.
</item 5>

<item 6>
Officer, Victoria
Officer is a satellite suburb of Melbourne, Victoria, Australia, 54 km south-east of Melbourne's Central Business District, located within the Shire of Cardinia
local government area.

 Officer recorded a population of 7,133 at the 2016 Census.
</item 6>

<item 7>
Electoral district of North Eastern Boroughs
North Eastern Boroughs was an electoral district of the Legislative Assembly in the Australian state of New South Wales from 1856 to 1859 that included the
towns of Newcastle, Stockton and Raymond Terrace.

 It was partly replaced by the electoral district of Newcastle and the electoral district of Hunter.
</item 7>

<item 8>
Electoral district of Tullamarine
Electoral district of Tullamarine was an electoral district of the Legislative Assembly in the Australian state of Victoria.

 The district centred on the suburb of Tullamarine, about 17km north-west of Melbourne.
</item 8>

<item 9>
Electoral district of Broken Hill
Broken Hill was an electoral district of the Legislative Assembly in the Australian state of New South Wales.

 From 1894 to 1913, the Broken Hill district was covered by the electoral districts of "Broken Hill", Willyama and Sturt.

 In the 1912 redistribution, the electoral district of "Broken Hill" was abolished and absorbed into "Willyama" and "Sturt".

 In 1920, "Willyama", "Sturt" and the electoral district of Cobar were combined to create a three-member "Sturt".

 In 1927, single-member electorates were recreated and the city of Broken Hill was split between "Sturt" and Murray.

 From 1932, the city of Broken Hill was split between "Sturt" and Cobar.

 In 1968 "Sturt" was renamed Broken Hill and included all of the city of Broken Hill.

 In 1999, it was merged with part of "Murray" and renamed the electoral district of Murray-Darling.
</item 9>

<item 10>
Electoral district of Nunawading
The electoral district of Nunawading was an electoral district of the Legislative Assembly in the Australian state of Victoria, located in the eastern Melbourne
suburb of Nunawading.

 A notable former member was future Prime Minister, Robert Menzies.
</item 10>
=== END ITEM id=38016e1ec0bc3a96 ===

=== ITEM id=4bd4e28c789a60f4 ===
Gold sentences and facts:
Sentence 0: David Michael Draiman (born March 13, 1973) is an American songwriter and the vocalist for the band Disturbed as well as for the band Device.
  Fact 0.0: David Michael Draiman was born on March 13, 1973.
  Fact 0.1: David Michael Draiman is an American songwriter.
  Fact 0.2: David Michael Draiman is the vocalist for the band Disturbed.
  Fact 0.3: David Michael Draiman is the vocalist for the band Device.
Sentence 1: Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and founding members of the rock band Chicago.
  Fact 1.0: Terry Alan Kath lived January 31, 1946 to January 23, 1978.
  Fact 1.1: Terry Alan Kath was an American musician.
  Fact 1.2: Terry Alan Kath was an American songwriter.
  Fact 1.3: Terry Alan Kath was best known as the original guitarist of the rock band Chicago.
  Fact 1.4: Terry Alan Kath was one of the lead singers of the rock band Chicago.
  Fact 1.5: Terry Alan Kath was a founding member of the rock band Chicago.

Retrieved items:
<item 1>
Device (metal band)
Device was an industrial metal band started by David Draiman, frontman of the heavy metal group Disturbed.

 Draiman was approached by Geno Lenardo, former guitarist of Filter, and together they started to work on new material.

 The two started work on a debut album in June 2012.

 The result of those sessions, "Device", was released on April 9, 2013.

 The first single and third track, "Vilify" was released to radio ahead of the album on February 19, 2013, alongside its first music video, directed by P. R.
Brown.

 The second single and the introductory track, "You Think You Know", was released on June 11, 2013, alongside its music video, directed once again by Brown.
</item 1>

<item 2>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 2>

<item 3>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 3>

<item 4>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 4>

<item 5>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 5>

<item 6>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 6>

<item 7>
Prayer (song)
"Prayer" is a song released on August 14, 2002 by the American heavy metal band Disturbed as the first single from their second album, "Believe".

 It was inspired by the death of vocalist David Draiman's grandfather as well as various circumstances after the September 11 attacks, and is about a
conversation between Draiman and God.

 Upon release, many media outlets refused to air the "Prayer" music video, citing supposed similarities between the imagery of the music video and that of the
September 11 attacks.

 "Prayer" peaked at number-three on two United States airplay charts, "Billboard"'s Mainstream Rock Tracks and Modern Rock Tracks charts, as well as peaking at
number-fifty-eight on "Billboard"'s Hot 100 and number-fourteen on the Canadian Singles Chart.

 "Prayer" is Disturbed's second highest charting single on the Billboard Hot 100 and their highest charting single on the Modern Rock Tracks chart, and 1 of
only 2 of their songs to reach the top 5 on the chart (the other being "Inside the Fire", which peaked at No. 4).
</item 7>

<item 8>
Pantera
Pantera was an American heavy metal band from Arlington, Texas.

 The group was formed in 1981 by the Abbott brothers – drummer Vinnie Paul and guitarist Dimebag Darrell – along with lead vocalist Terry Glaze.

 Bassist Rex Brown joined the band the following year, replacing Tommy D. Bradford, who was the unofficial original.

 Having started as a glam metal band, Pantera released four albums during the 1980s.

 Looking for a new and heavier sound, Pantera replaced Glaze with Phil Anselmo in late 1986 and released "Power Metal" in 1988.

 With its fifth album, 1990's "Cowboys from Hell", Pantera introduced a groove metal sound.

 Pantera's sixth album, 1992's "Vulgar Display of Power", exhibited an even heavier sound. "

Far Beyond Driven" (1994) debuted at number one on the "Billboard" 200.
</item 8>

<item 9>
David Draiman
David Michael Draiman (born March 13, 1973) is an American songwriter and the vocalist for the band Disturbed as well as for the band Device.

 Draiman is known for his distorted voice and percussive singing style.

 In November 2006, Draiman was voted number 42 on the "Hit Parader"’s ""Top 100 Metal Vocalists of All Time"".

 Draiman has written some of Disturbed's most successful singles, such as "Stupify", "Down with the Sickness", "Indestructible", and "Inside the Fire".
</item 9>

<item 10>
Wishing You Were Here
"Wishing You Were Here" is a song written by Peter Cetera for the group Chicago and recorded for their album "Chicago VII" (1974), with lead vocals by Terry
Kath (uncredited on the original album package), while Cetera sang the song's bridge.

 The third single released from that album, it reached #11 on the U.S. "Billboard" Hot 100, #9 on the "Cash Box" Top 100, and hit #1 on the Easy Listening
chart.
</item 10>
=== END ITEM id=4bd4e28c789a60f4 ===

=== ITEM id=a241b5140f658157 ===
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
Cracks (film)
Cracks is a 2009 independent drama psychological thriller film starring Eva Green, James McGovern, Juno Temple, María Valverde, and Imogen Poots, which was
released theatrically in the UK and Ireland on 4 December 2009.

 In the United States it was released by IFC Films on 18 March 2011, and premiered on Showtime in late 2011.
</item 1>

<item 2>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 2>

<item 3>
Juliet Weber
Juliet Weber is a producer, director, writer, and an editor for the film industry.

 She co-produced films such as Fastpitch (2000) and Unchained Memories (2003).
</item 3>

<item 4>
Magic Magic (2013 film)
Magic Magic is a 2013 American-Chilean psychological thriller film written and directed by Sebastián Silva and starring Juno Temple, Emily Browning, Michael
Cera, and Catalina Sandino Moreno.
</item 4>

<item 5>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
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
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 8>

<item 9>
Lovelace (film)
Lovelace is a 2013 American biographical drama film about porn actress Linda Boreman, better known as Linda Lovelace, star of "Deep Throat", a seminal 1972 film
at the forefront of the Golden Age of Porn.

 "Lovelace" covers her life from age 20 to 32.

 Directed by Rob Epstein and Jeffrey Friedman, the film was written by Andy Bellin and stars Amanda Seyfried, Peter Sarsgaard, Sharon Stone, Adam Brody, James
Franco, Chloë Sevigny, and Juno Temple.

 The film had its world premiere on January 22, 2013, at the 2013 Sundance Film Festival and opened in a U.S. limited release on August 9, 2013.
</item 9>

<item 10>
Dirty Girl (2010 film)
Dirty Girl is a 2010 American coming of age comedy-drama film written and directed by Abe Sylvia.

 It stars Juno Temple, Milla Jovovich and William H. Macy.

 It premiered at the 2010 Toronto International Film Festival on September 12, 2010.

 It was distributed theatrically by The Weinstein Company on October 7, 2011.
</item 10>
=== END ITEM id=a241b5140f658157 ===

=== ITEM id=d3667816a156be23 ===
Gold sentences and facts:
Sentence 0:  In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
  Fact 0.0: In Crash, there is no betting, unlike in Brag.
  Fact 0.1: In Crash, players aim to reach a total of 11 points.
  Fact 0.2: In Crash, the 11 points are gained over successive deals.
Sentence 1: Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.
  Fact 1.0: Three-card brag is a 16th-century British card game.
  Fact 1.1: Three-card brag is the British national representative of the vying or 'bluffing' family of gambling games.

Retrieved items:
<item 1>
Three card brag
Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.

 Brag is a direct descendant of the Elizabethan game of Primero and one of the several ancestors to poker, just varying in betting style and hand rankings.
</item 1>

<item 2>
Quadrille (card game)
Quadrille is a card game that was popular in the 18th century.

 A variant of the Spanish card game Ombre, it is played by four players in pairs, with a deck of 40 cards (the 8's, 9's and 10's being removed).

 By the end of the 19th century, the card game had fallen out of fashion.
</item 2>

<item 3>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 3>

<item 4>
Egyptian Ratscrew
Egyptian Ratscrew (also known as Egyptian Ratkiller, Egyptian War, and by other names) is a card game of the matching family of games.

 The game is similar to the 19th century British card game Beggar-My-Neighbour, with the added concept of "slapping" cards when certain combinations are played,
similar to and perhaps borrowed from Slapjack.
</item 4>

<item 5>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 5>

<item 6>
Put (card game)
Put is an English tavern trick-taking card game first recorded in the 16th century and later castigated by 17th century moralists as one of ill repute.

 It belongs to a very ancient family of card games and clearly relates to a group known as Trut, "Truque", also "Tru", and the South American game Truco.

 Its more elaborate version is the Spanish game of Truc, which is still much played in many parts of Southern France and Spain.
</item 6>

<item 7>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 7>

<item 8>
Ombre
The historical importance of Ombre in the field of playing cards is the fact that it was the first card game in which a trump suit was established by bidding
rather than by the random process of turning the first card of the stock.

 This game developed from Triunfo, though it was from L'Hombre that the idea of bidding was adopted into other card games such as Skat, and Tarot, which owes
Hombre a good portion of its betting system as well.

 The game continued to be in vogue almost in every corner of Europe from the late 17th through the 18th centuries.
</item 8>

<item 9>
Triomphe
Triomphe (French for triumph) is a card game dating from the late 15th century.

 It most likely originated in France or Spain (as triunfo) and later spread to the rest of Europe.

 When the game arrived in Italy, it shared a similar name with the pre-existing game and deck known as "trionfi" (tarot).

 While trionfi has a fifth suit that acts as permanent trumps, triomphe randomly selects one of the existing four suits as trumps.

 Another common feature of this game is the robbing of the stock.

 Triomphe became so popular that during the 16th century the earlier game of trionfi was gradually renamed tarocchi, tarot, or tarock.

 This game is the origin of the English word "trump" and is the ancestor of many trick-taking games like Euchre (via Écarté) and Whist (via Ruff and Honours).
</item 9>

<item 10>
Crash (card game)
Crash is a British card game extension of Nine-card Brag.

 In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
</item 10>
=== END ITEM id=d3667816a156be23 ===

=== ITEM id=4cdc881db0f07f8e ===
Gold sentences and facts:
Sentence 0: The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.
  Fact 0.0: The Sports Museum is also known as The Sports Museum of New England.
  Fact 0.1: The Sports Museum is a non-profit museum.
  Fact 0.2: The Sports Museum is currently located in the TD Garden.
  Fact 0.3: The TD Garden is in Boston, Massachusetts.
Sentence 1: TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.
  Fact 1.0: TD Garden is often called Boston Garden.
  Fact 1.1: TD Garden is often called "The Garden".
  Fact 1.2: TD Garden is a multi-purpose arena.
  Fact 1.3: TD Garden is in Boston, Massachusetts.
Sentence 2:  It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
  Fact 2.0: TD Garden opened in 1995.
  Fact 2.1: TD Garden opened as a replacement for the original Boston Garden.
  Fact 2.2: TD Garden has been known as Shawmut Center.
  Fact 2.3: TD Garden has been known as FleetCenter.
  Fact 2.4: TD Garden has been known as TD Banknorth Garden.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 2>

<item 3>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 3>

<item 4>
Estonian Sports Museum
The Estonian Sports and Olympic Museum, founded in 1963 and modernised in 2001, is the largest sports museum in the Baltic states.

 The museum is located on Rüütli street in Tartu, Estonia and the museum has a small outpost in Otepää that concentrates on the Winter sports.

 Before 2016, Estonian Sports and Olympic Museum was named Estonian Sports Museum.
</item 4>

<item 5>
Smith Fieldhouse
The George Albert Smith Fieldhouse is a 5,000 seat multi-purpose arena in Provo, Utah.

 Built in 1951, it is the home of the Brigham Young University Cougars volleyball teams and most home gymnastics meets.

 It was named for George Albert Smith, the eighth president of The Church of Jesus Christ of Latter-day Saints, who died the year the fieldhouse opened.

 Prior to the Marriott Center opening in 1971 it was home to the basketball teams.

 At that time, the arena held 10,500 people.

 Smith Fieldhouse also has a track and several offices used by BYU's athletic department.
</item 5>

<item 6>
Sports Museum of America
The Sports Museum of America (SmA) was the United States' first national sports museum dedicated to the history and cultural significance of sports in America.

 It opened in May 2008 and closed less than nine months later, in February 2009.
</item 6>

<item 7>
Sports Legends Museum at Camden Yards
Sports Legends Museum at Camden Yards was a non-profit sports museum in Baltimore, Maryland, United States, owned and operated by the Babe Ruth Birthplace &
Museum.

 It opened on May 14, 2005, with the daughter of celebrated baseball player Babe Ruth in attendance.

 After ten years of operation the museum closed abruptly on October 12, 2015 after failing to reach an agreement with the Maryland Stadium Authority for the
continued use of Camden Station.

 The 22000 sqft museum was adjacent to the main gate of Oriole Park at Camden Yards and has artifacts and interactive exhibits profiling Maryland’s sports
history.

 Exhibits included such area teams as the Baltimore Orioles, Baltimore Ravens, Baltimore Colts, Maryland Terrapins, Baltimore Elite Giants, Baltimore Black Sox,
and the Baltimore Blast.
</item 7>

<item 8>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 8>

<item 9>
Fanattic Sports Museum
Fanattic Sports Museum (FSM) is a sports museum, dedicated to various types of sports and located in Ecospace Business Park, New Town, Kolkata, India.

 Established in January 2017, FSM is known as the first sports museum in Kolkata as well as in West Bengal.
</item 9>

<item 10>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 10>
=== END ITEM id=4cdc881db0f07f8e ===

=== ITEM id=b20448a1dd5046e7 ===
Gold sentences and facts:
Sentence 0: Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.
  Fact 0.0: Innocence is a 2013 American horror drama film.
  Fact 0.1: Innocence was directed by Hilary Brougher.
  Fact 0.2: Hilary Brougher co-wrote Innocence with Tristine Skyler.
Sentence 1:  The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
  Fact 1.0: The movie Innocence stars Sophie Curtis.
  Fact 1.1: The movie Innocence stars Kelly Reilly.
  Fact 1.2: The movie Innocence stars Graham Phillips.
  Fact 1.3: The movie Innocence stars Linus Roache.
  Fact 1.4: The movie Innocence stars Sarah Sutherland.
  Fact 1.5: The movie Innocence stars Stephanie March.
Sentence 2: Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.
  Fact 2.0: Stephanie Caroline March was born on 23, 1974.
  Fact 2.1: Stephanie Caroline March is an American actress.
Sentence 3:  She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
  Fact 3.0: Stephanie March is best known for her role as Alexandra Cabot.
  Fact 3.1: The role of Alexandra Cabot was in a long-running NBC series.

Retrieved items:
<item 1>
Kim Johnston Ulrich
Kim Johnston Ulrich (born March 24, 1955) is an American actress.

 From 1983 to 1986, Ulrich played the role of Diana McColl on "As the World Turns".

 In 1988, she guest-starred in two episodes of "Werewolf".

 In 1990, Ulrich appeared on the TV series "Wings" as Carol, Brian's ex-wife.

 In 1993, Ulrich appeared in the pilot episode of "" as Dr. Antoinette Baines, a villainous scientist.

 She appeared in a third season episode of "Highlander: The Series" in 1995.

 She appeared in the first episode of season 4 of "" as a conniving wife in "Murder by Friendly Fire" in 1996.
</item 1>

<item 2>
Gloria Garayua
Gloria Garayua (born October 18, 1978 in New York City) is an American film and television actress.

 Garayua made her major film debut in the 2005 comedy film "Fun With Dick and Jane", and is now commonly cast in guest roles on long-running series such as "Six
Feet Under", "Weeds" and "The Shield".

 After being cast in an ongoing role in "Grey's Anatomy", Garayua has played recurring roles in other popular TV shows such as "Cougar Town" and "How to Get
Away with Murder".
</item 2>

<item 3>
Conviction (2006 TV series)
Conviction is an American television drama on NBC that debuted as a mid-season replacement on Friday, March 3, 2006.

 The cast includes Stephanie March reprising her "" role as Alexandra Cabot.

 In the series, Cabot returns to New York City and becomes a Bureau Chief ADA supervising a group of young but talented assistant district attorneys after a
stint in the Witness Protection Program.

 Other cast members include Eric Balfour, Anson Mount, Jordan Bridges, Julianne Nicholson, Milena Govich, and J. August Richards.
</item 3>

<item 4>
Rebecca Forstadt
Rebecca Lynn Forstadt (born December 16, 1953), also known as Reba West, is an American voice actress, best known for playing young female roles in various
animated series.

 After studying theater at Orange Coast College, in Costa Mesa, California, Forstadt began her acting career by working at Knott's Berry Farm's Bird Cage
Theater, performing melodramas, often as the damsel in distress character.

 Later, she went to Hollywood where she worked as a wardrobe mistress on such television shows as "The White Shadow" and "Hill Street Blues", as well as for the
film "S.O.B.".

 She also spent several years doing live theater in the Los Angeles area.

 Most notably, she won some recognition for her portrayal of the character Josette in the world premiere of Eugène Ionesco's "Tales for People Under 3 Years of
Age" at the Stages Theatre Center in 1982.

 She starred in several low-budget movies such as "Mugsy's Girls", with Ruth Gordon and Laura Branigan, and "Round Numbers" with Kate Mulgrew, Samantha Eggar,
and Shani Wallis.
</item 4>

<item 5>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 5>

<item 6>
Allison Miller
Allison Miller (born September 2, 1985) is an American actress.

 She is best known for playing Michelle Benjamin on the NBC series "Kings", Skye Tate on the Fox series "Terra Nova", and Carrie on the NBC series "Go On".

 She starred as Laura Larson on the Syfy television series "Incorporated".
</item 6>

<item 7>
Anastasia Horne
Anastasia Nicole "Ana" Horne (born August 5, 1978 in Claremont, California) is a former American actress and singer who has appeared on television and in the
theater.

 Her last television role was "Lori" on the MTV series "Undressed" in 2000.

 She also starred as "Lark Madison-Scanlon" on the daytime soap opera "Port Charles" from 1997 to 1999.

 As a teenager, she was cast as "Ana" on the long-running NBC/The Disney Channel series "Kids Incorporated" from 1984, to 1985, and 1991 to the end of the
series in 1994.
</item 7>

<item 8>
Stephanie March
Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.

 She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
</item 8>

<item 9>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 9>

<item 10>
Innocence (2013 film)
Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.

 The movie is based on the 2000 novel of the same name by Jane Mendelsohn.

 It had its world premiere on 26 October, 2013 at the Austin Film Festival and received a limited theatrical release in the United States on 5 September, 2014.

 The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
</item 10>
=== END ITEM id=b20448a1dd5046e7 ===
