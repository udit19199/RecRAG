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

=== ITEM id=0090fe6fac725141 ===
Gold sentences and facts:
Sentence 0: In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
  Fact 0.0: The Fader published an interview with CupcakKe in September.
  Fact 0.1: In the interview, CupcakKe discussed her new album.
  Fact 0.2: CupcakKe's new album would address "rape, and everything else".
Sentence 1: The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.
  Fact 1.0: The Fader, stylized as The FADER, is a New York City-based music magazine.
  Fact 1.1: The Fader was launched in 1999.
  Fact 1.2: The Fader was launched by Rob Stone and Jon Cohen.
  Fact 1.3: The Fader covers music, style and culture.
Sentence 2:  It was the first print publication to be released on iTunes.
  Fact 2.0: The Fader was the first print publication to be released on iTunes.

Retrieved items:
<item 1>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 1>

<item 2>
Audacious (CupcakKe album)
In September, The Fader published an interview with CupcakKe, which she discussed her new album that would address "rape, and everything else".
</item 2>

<item 3>
Sculptural Pursuit
Sculptural Pursuit was a quarterly art/literary magazine published by Hammer & Pen Productions, a Denver, Colorado publishing company.

 The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
</item 3>

<item 4>
Exclaim!
Exclaim!

 is a monthly Canadian music magazine that features in-depth coverage of new music across all genres with a special focus on Canadian and cutting-edge artists.

 Content is based on the monthly print publication, which publishes 9 issues per year, distributing over 103,000 copies to over 2,600 locations across Canada.

 The magazine has an average of 361,200 monthly readers.

 Their website, exclaim.ca, has an average of 675,000 unique visitors a month.
</item 4>

<item 5>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 5>

<item 6>
State (magazine)
State.ie (formerly State Magazine) is an Irish website and formerly a monthly music magazine, which launched in April 2008 and ceased to print in January 2009
having published a total of ten issues.

 The magazine continues online and was voted Best Music Site in 2008 and Best Web Publication in 2010 in the Irish Web Awards.

 Originally the concept of the magazine involved a hard copy of which there was a charge to purchase, however after six issues it was decided to make the
magazine's content free both online and in print.

 The first issue, April 2008, appeared on Irish shelves on 6 March 2008 and featured Michael Stipe of R.E.M. on the cover.

 This immediately garnered comparisons between the new magazine's similarities with "Hot Press" who featured Stipe on their cover at the same time, a move
widely thought to be an attempt by Hot Press to stifle State's status as a serious 'alternative' to the more established local magazine.

 At a price of €5.50, "State" charged €2 more than "Hot Press".
</item 6>

<item 7>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 7>

<item 8>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 8>

<item 9>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 9>

<item 10>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 10>
=== END ITEM id=0090fe6fac725141 ===

=== ITEM id=711854eb5a866453 ===
Gold sentences and facts:
Sentence 0: Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.
  Fact 0.0: Bernard Malamud was born on April 26, 1914.
  Fact 0.1: Bernard Malamud died on March 18, 1986.
  Fact 0.2: Bernard Malamud was an American novelist.
  Fact 0.3: Bernard Malamud was an American short story writer.
Sentence 1: Lois McMaster Bujold ( ; born November 2, 1949) is an American speculative fiction writer.
  Fact 1.0: Lois McMaster Bujold was born on November 2, 1949.
  Fact 1.1: Lois McMaster Bujold is an American speculative fiction writer.

Retrieved items:
<item 1>
Gentleman Jole and the Red Queen
Gentleman Jole and the Red Queen is a science fiction novel by Lois McMaster Bujold.

 It is an installment in Bujold's Vorkosigan Saga.

 Bujold has described it as "not a war story.

 It is about grownups."
</item 1>

<item 2>
Brothers in Arms (Bujold novel)
Brothers in Arms is a science fiction novel by American writer Lois McMaster Bujold, part of the Vorkosigan Saga.

 It was the fifth book published in the series, and is the twelfth story, including novellas, in the internal chronology of the series.

 "Brothers in Arms" was first published by Baen Books in January 1989, and is included in the 2002 omnibus "Miles Errant".
</item 2>

<item 3>
Passage (Bujold novel)
Passage is a novel by American writer Lois McMaster Bujold, published in 2008.

 It is the third in the tetralogy "The Sharing Knife".
</item 3>

<item 4>
Shards of Honor
Shards of Honor is an English language science fiction novel by Lois McMaster Bujold, first published in June 1986.

 It is a part of the Vorkosigan Saga, and is the first full-length novel in publication order.

 "Shards of Honor" is paired with Bujold's 1991 "Barrayar" in the omnibus "Cordelia's Honor" (1996).
</item 4>

<item 5>
Ethan of Athos
Ethan of Athos is a 1986 science fiction novel by American author Lois McMaster Bujold.

 The title character is Dr. Ethan Urquhart, Chief of Biology at the Severin District Reproduction Centre on the planet Athos, who is sent to find out what
happened to a shipment of vital ovarian tissue cultures.

 Set in the fictional universe of Bujold's Vorkosigan Saga, the novel mentions but does not feature her usual protagonist Miles Vorkosigan.

 To date, Bujold has never revisited the settings of Athos or Kline Station in her many subsequent novels, but the events of "Ethan of Athos" are later referred
to indirectly in the novels "Borders of Infinity" (1989) and "Cetaganda" (1995).
</item 5>

<item 6>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 6>

<item 7>
Lois McMaster Bujold
Lois McMaster Bujold ( ; born November 2, 1949) is an American speculative fiction writer.

 She is one of the most acclaimed writers in her field, having won the Hugo Award for best novel four times, matching Robert A. Heinlein's record, not counting
his Retro Hugo.

 Her novella "The Mountains of Mourning" won both the Hugo Award and Nebula Award.

 In the fantasy genre, "The Curse of Chalion" won the Mythopoeic Award for Adult Literature and was nominated for the 2002 World Fantasy Award for best novel,
and both her fourth Hugo Award and second Nebula Award were for "Paladin of Souls".

 In 2011 she was awarded the Skylark Award.

 In 2013 she was awarded the Forry Award.

 In 2017 she won a Hugo Award for Best Series, for the Vorkosigan Saga.
</item 7>

<item 8>
Cryoburn
Cryoburn is a science fiction novel by Lois McMaster Bujold, first published in October 2010.

 Part of the Vorkosigan Saga, it was nominated for the Hugo Award for Best Novel in 2011, as Bujold's ninth Best Novel nomination.

 Also in 2011, it was one of the top five finishers in the poll for the Locus Award for Best Science Fiction Novel.
</item 8>

<item 9>
Horizon (novel)
Horizon is a fantasy novel by American writer Lois McMaster Bujold.

 It is the fourth in the tetralogy "The Sharing Knife".
</item 9>

<item 10>
Barrayar
Barrayar is a science fiction novel by Lois McMaster Bujold.

 It was first published as four installments in "Analog" in July–October 1991, and then published in book form by Baen Books in October 1991.

 "Barrayar" won both the Hugo Award for Best Novel and the Locus Award for Best Science Fiction Novel in 1992.

 It is a part of the Vorkosigan Saga, and is the seventh full-length novel of the series, in publication order.

 "Barrayar" is a direct sequel to Bujold's first novel, "Shards of Honor" (1986), and the two are paired in the 1996 omnibus "Cordelia's Honor".
</item 10>
=== END ITEM id=711854eb5a866453 ===

=== ITEM id=fedec97754a024c2 ===
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
Canada under British rule
Although confederation in 1867 led to an enlarged Dominion with increased autonomy over domestic affairs, Canada still remained a colony within the British
Empire and was thus subordinate to the British Parliament until the enactment of the Statute of Westminster in 1931.

 This statute recognized Canada as an independent peer coequal with the United Kingdom, and thus provided the Parliament of Canada with legislative sovereignty
over all federal matters except the power to change the constitutional laws of Canada which remained under the purview of the Parliament of the United Kingdom.

 Canada's final vestige of legal dependence on the United Kingdom was terminated in 1982 with the enactment of the Canada Act, subsequently providing Canada
with full legal sovereignty completely independent of the United Kingdom.
</item 1>

<item 2>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 2>

<item 3>
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
</item 3>

<item 4>
House of Saxe-Coburg and Gotha
The House of Saxe-Coburg and Gotha ( ; German: "") is a German dynasty that ruled the duchy of Saxe-Coburg and Gotha, which was one of the Ernestine duchies.

 It is a cadet branch of the Saxon House of Wettin.
</item 4>

<item 5>
Succession to the British throne
Succession to the British throne is determined by descent, gender (for people born before October 2011), legitimacy, and religion.

 Under common law, the Crown is inherited by a sovereign's children or by a childless sovereign's nearest collateral line.

 The Bill of Rights 1689 and the Act of Settlement 1701, restrict succession to the throne to the legitimate Protestant descendants of Sophia of Hanover that
are in "communion with the Church of England".

 Spouses of Roman Catholics were disqualified from 1689 until the law was amended in 2015.

 Protestant descendants of those excluded for being Roman Catholics are eligible.
</item 5>

<item 6>
House of Oldenburg
The House of Oldenburg is a European dynasty of North German origin.

 It is one of Europe's most influential royal houses, with branches that rule or have ruled in Denmark, Iceland, Greece, Norway, Russia, Sweden, Schleswig,
Holstein, and Oldenburg.

 The current Queen of Denmark and King of Norway, the former King of Greece, the consorts of Greece and the United Kingdom, as well as the first twelve names in
the line of succession to the British throne, are all patrilineal members of the Glücksburg branch of this house.
</item 6>

<item 7>
Principality of Calenberg
The Principality of Calenberg was a dynastic division of the Welf duchy of Brunswick-Lüneburg established in 1432.

 Calenberg was ruled by the House of Hanover from 1635 onwards; the princes received the ninth electoral dignity of the Holy Roman Empire in 1692.

 Their territory became the nucleus of the Electorate of Hanover, ruled in personal union with the Kingdom of Great Britain from 1714 onwards.

 The principality received its name from Calenberg Castle, a residence of the Brunswick dukes.
</item 7>

<item 8>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 8>

<item 9>
House of Hanover
The House of Hanover (or the Hanoverians ; German: "Haus Hannover" ) is a German royal dynasty that ruled the Electorate and then the Kingdom of Hanover, and
that also provided monarchs of Great Britain and Ireland from 1714 and ruled the United Kingdom until the death of Queen Victoria in 1901.

 Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
</item 9>

<item 10>
Duke of Bedford
He regained the earldom in 1485 when his nephew Henry VII came to the throne and was elevated to the dukedom the same year.

 He had no legitimate children and the titles became extinct on his death in 1495.
</item 10>
=== END ITEM id=fedec97754a024c2 ===

=== ITEM id=5879c9c67b107b90 ===
Gold sentences and facts:
Sentence 0:  He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
  Fact 0.0: He received an excellent education in Arabic.
  Fact 0.1: He received an excellent education in Greek.
  Fact 0.2: He studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.
Sentence 1: Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.
  Fact 1.0: Abul ʿAla Al-Maʿarri's Arabic name is أبو العلاء المعري (Abū al-ʿAlāʾ al-Maʿarrī).
  Fact 1.1: Abul ʿAla Al-Maʿarri's full name is أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري (Abū al-ʿAlāʾ Aḥmad ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī).
  Fact 1.2: Abul ʿAla Al-Maʿarri was born in December 973.
  Fact 1.3: Abul ʿAla Al-Maʿarri died in May 1057.
  Fact 1.4: Abul ʿAla Al-Maʿarri was blind.
  Fact 1.5: Abul ʿAla Al-Maʿarri was an Arab philosopher.
  Fact 1.6: Abul ʿAla Al-Maʿarri was an Arab poet.
  Fact 1.7: Abul ʿAla Al-Maʿarri was an Arab writer.

Retrieved items:
<item 1>
Ibn al-Khatib
Lisan ad-Din ibn al-Khatib (Born 16 November 1313, Loja– died 1374, Fes, Morocco) (Full name Muhammad ibn Abd Allah ibn Said ibn Ali ibn Ahmad al-Salmani) was a
Moroccan Arab polymath poet, writer, historian, philosopher, physician and politician from Emirate of Granada.

 Some of his poems decorate the walls of the Alhambra in Granada.
</item 1>

<item 2>
Eenasul Fateh (Aladin, b. 1959) is a Bangladeshi-British magician and cultural practitioner. “Abdāl” denotes saintly figures in Islamic mysticism. Dawud Tai (d.
777–782) was a Sufi scholar; Muhammad ibn Thaleb a 15th-century Persian physician. Ibn al-Khatib (1313–1374) was a Granada-born Moroccan polymath. Da’ud Abu
</item 2>

<item 3>
Al-Mutanabbi
Al-Mutanabbi had a great pride in himself through his poetry.

 Among the topics he discussed were courage, the philosophy of life, and the description of battles.

 Many of his poems were and still are widely spread in today's Arab world and are considered to be proverbial.
</item 3>

<item 4>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 4>

<item 5>
Shaykh Muhammad ibn Thaleb
Muhammad ibn Thalib ibn Abd Allah ibn Ni`mat Allah ibn Sadr ad-Din ibn Shaykh Baha' ad-Din ash-Shirazi (Arabic: محمد ابن طالب عبدالله ابن نعمات الله ابن صدر
الدين ابن شيخ بهاءالدين الشيرازي‎ ‎ ) was a 15th-century Persian physician from Shiraz, Iran.
</item 5>

<item 6>
The entries cover major Arab cultural figures and works: Egyptian actor Ahmed Zaki and director Salah Abu Seif, known for intense performances and Egyptian
neorealism; and writer Naguib Mahfouz, Nobel laureate whose novels include *The Cairo Trilogy*, *The Beginning and the End*, *Thebes at War* and *Arabian Nights
and Days*. Abu Seif’s 1960 *Bidaya wa Nihaya* adapted Mahfouz. Other topics include Tayeb
</item 6>

<item 7>
Abd Allah ibn Abbas
Abd Allah ibn Abbas (Arabic: عبد الله ابن عباس‎ ‎ ) or ′Abd Allah ibn al-′Abbas otherwise called (Ibn Abbas; Al-Habr; Al-Bahr; The Doctor; The Sea) was born c.
619 CE.

 He was the son of Al-‘Abbas ibn ‘Abd al-Muttalib, an uncle of the Islamic prophet Muhammad, and a nephew of the Maymunah bint al-Harith, who later became
Muhammad's wife.

 He was one of Muhammad's cousins and one of the early Qur'an scholars.
</item 7>

<item 8>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 8>

<item 9>
Al-Maʿarri
Abul ʿAla Al-Maʿarri (Arabic أبو العلاء المعري "Abū al-ʿAlāʾ al-Maʿarrī ", full name أبو العلاء أحمد بن عبد الله بن سليمان التنوخي المعري "Abū al-ʿAlāʾ Aḥmad
ibn ʿAbd Allāh ibn Sulaimān al-Tanūẖī al-Maʿarrī "; December 973 – May 1057) was a blind Arab philosopher, poet, and writer.

 Al-Maʿarri held and expressed an irreligious world view which was met with controversy, but in spite of it, he is regarded as one of the greatest classical
Arabic poets.
</item 9>

<item 10>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 10>
=== END ITEM id=5879c9c67b107b90 ===

=== ITEM id=64b95b61a90bc2d9 ===
Gold sentences and facts:
Sentence 0:  Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
  Fact 0.0: Dan Hurley is the son of Bob Hurley.
  Fact 0.1: Bob Hurley is a Hall of Fame high school coach.
  Fact 0.2: Dan Hurley is the younger brother of Bobby Hurley.
  Fact 0.3: Bobby Hurley is a former Duke guard.
  Fact 0.4: Bobby Hurley is a former Sacramento Kings guard.
Sentence 1: Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.
  Fact 1.0: Robert Matthew Hurley was born on June 28, 1971.
  Fact 1.1: Robert Matthew Hurley is an American basketball coach.
  Fact 1.2: Robert Matthew Hurley is a former college player.
  Fact 1.3: Robert Matthew Hurley is a former professional player.
Sentence 2:  Hurley is currently the head coach of the Arizona State men's team.
  Fact 2.0: Hurley is currently the head coach of the Arizona State men's team.

Retrieved items:
<item 1>
2011–12 Wagner Seahawks men's basketball team
The 2011–12 Wagner Seahawks men's basketball team represented Wagner College during the 2011–12 NCAA Division I men's basketball season.

 The Seahawks, led by second year head coach Dan Hurley, played their home games at Spiro Sports Center and are members of the Northeast Conference.

 They finished the season 25–6, 15–3 in NEC play to finish in second place.

 They lost in the semifinals of the NEC Basketball Tournament to Robert Morris.

 Despite having 25 wins, the Seahawks did not accept an invitation to a post season tournament.
</item 1>

<item 2>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 2>

<item 3>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 3>

<item 4>
Rhode Island Rams men's basketball
The Rhode Island Rams men's basketball team is a college basketball program that competes in NCAA Division I and the Atlantic 10 Conference representing the
University of Rhode Island.

 The team is under the direction of head coach Dan Hurley, who was hired on March 20, 2012.

 The Rams play their home games at the Ryan Center (capacity 7,657).
</item 4>

<item 5>
Rudy Lavik
Rudolph H. "Rudy" Lavik (April 30, 1892 – September 29, 1979) was an American football and basketball coach and college athletics administrator.

 He served as the head football coach at Concordia College in Moorhead, Minnesota from 1920 to 1921, at Arizona State Teacher's College of Flagstaff—now
Northern Arizona University—from 1928 to 1932, and at Arizona State Teachers College at Tempe—now Arizona State University—from 1933 to 1937, compiling a career
college football record of 37–42–7.

 Lavik was also the head basketball coach at Arizona State Flagstaff (1927–1931) and Arizona State Tempe (1933–1935, 1939–1948), tallying a career college
basketball mark of 146–137.

 In addition he served as the athletic director at Arizona State from 1933 to 1949.

 He was a graduate of Springfield College in Springfield, Massachusetts.
</item 5>

<item 6>
Bob Hurley
Robert Emmet Hurley, Sr. (born July 31, 1947) is a basketball coach at St. Anthony High School in Jersey City, New Jersey.

 Hurley has amassed 28 state championships and more than 1000 wins in 39 years as a coach.

 On February 2, 2011, Hurley became the tenth coach in high school history to win 1000 games.

 He is featured in the documentary "The Street Stops Here".

 Five of his teams have gone undefeated.

 On April 5, 2010, he was announced as the only coach to be inducted into the Naismith Memorial Basketball Hall of Fame that year and only the third high school
coach in history to be so honored; he was formally inducted on August 13 of that year.

 Hurley is the father of Bobby Hurley, a former All-American point guard at Duke and the head basketball coach at Arizona State, and Dan Hurley, who was hired
in February 2012 to coach the University of Rhode Island after two years of coaching at Wagner College and nine years coaching at Newark's Saint Benedict's
Preparatory School, also one of the top high school programs in the nation.
</item 6>

<item 7>
Nate Oats
Nathanael "Nate" Oats (born October 13, 1974) is an American basketball coach.

 He is currently the head basketball coach at the State University of New York at Buffalo, following two seasons as an assistant coach under Bobby Hurley.

 Oats was named head coach on April 11, 2015 after Hurley was hired by Arizona State as head coach.

 In only his first season as head coach, he took the Buffalo Bulls to their second straight NCAA Tournament bid after winning the Mid-American Conference
Championship.
</item 7>

<item 8>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 8>

<item 9>
2012–13 Wagner Seahawks men's basketball team
The 2012–13 Wagner Seahawks men's basketball team represented Wagner College during the 2012–13 NCAA Division I men's basketball season.

 The Seahawks were led by the youngest men's head coach in NCAA Division I, Bashir Mason, who was 28 when he was elevated from an assistant position in March
2012 following the departure of Dan Hurley for Rhode Island.

 The Seahawks played their home games at Spiro Sports Center and were members of the Northeast Conference.

 They finished the season 19–12, 12–6 in NEC play to finish in a three way tie for second place.

 They advanced to the semifinals of the Northeast Tournament where they lost to Long Island.

 For the second consecutive year, despite a winning record, Wagner choose not to participate in a post season tournament.
</item 9>

<item 10>
Dan Hurley
Daniel S. Hurley (born January 16, 1973) is currently the head basketball coach at the University of Rhode Island.

 He was named head coach on March 20, 2012 after a two-year stint at Wagner College.

 Prior to Wagner, Hurley had a standout career as head coach of Saint Benedict's Preparatory School, where he built the New Jersey school into one of the top
high school basketball programs in America.

 Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
</item 10>
=== END ITEM id=64b95b61a90bc2d9 ===

=== ITEM id=f803f756a5c873c0 ===
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
Electoral district of North Eastern Boroughs
North Eastern Boroughs was an electoral district of the Legislative Assembly in the Australian state of New South Wales from 1856 to 1859 that included the
towns of Newcastle, Stockton and Raymond Terrace.

 It was partly replaced by the electoral district of Newcastle and the electoral district of Hunter.
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
Officer, Victoria
Officer is a satellite suburb of Melbourne, Victoria, Australia, 54 km south-east of Melbourne's Central Business District, located within the Shire of Cardinia
local government area.

 Officer recorded a population of 7,133 at the 2016 Census.
</item 3>

<item 4>
Pakenham, Victoria
Pakenham is a satellite suburb of Melbourne on the edge of the West Gippsland region of Victoria, Australia, 56 km south-east of Melbourne's Central Business
District, located within the Shire of Cardinia local government area.

 Pakenham recorded a population of 46,421 at the 2016 Census.
</item 4>

<item 5>
Electoral district of Tullamarine
Electoral district of Tullamarine was an electoral district of the Legislative Assembly in the Australian state of Victoria.

 The district centred on the suburb of Tullamarine, about 17km north-west of Melbourne.
</item 5>

<item 6>
Electoral district of Broken Hill
Broken Hill was an electoral district of the Legislative Assembly in the Australian state of New South Wales.

 From 1894 to 1913, the Broken Hill district was covered by the electoral districts of "Broken Hill", Willyama and Sturt.

 In the 1912 redistribution, the electoral district of "Broken Hill" was abolished and absorbed into "Willyama" and "Sturt".

 In 1920, "Willyama", "Sturt" and the electoral district of Cobar were combined to create a three-member "Sturt".

 In 1927, single-member electorates were recreated and the city of Broken Hill was split between "Sturt" and Murray.

 From 1932, the city of Broken Hill was split between "Sturt" and Cobar.

 In 1968 "Sturt" was renamed Broken Hill and included all of the city of Broken Hill.

 In 1999, it was merged with part of "Murray" and renamed the electoral district of Murray-Darling.
</item 6>

<item 7>
Electoral district of Emerald Hill
Emerald Hill was an electoral district of the Victorian Legislative Assembly, Australia.

 It covered part of the inner-city suburb South Melbourne and consisted part of the previous Electoral district of South Melbourne which was abolished in 1859.

 (The other part of the South Melbourne electorate became the Electoral district of Sandridge).
</item 7>

<item 8>
Electoral district of Nunawading
The electoral district of Nunawading was an electoral district of the Legislative Assembly in the Australian state of Victoria, located in the eastern Melbourne
suburb of Nunawading.

 A notable former member was future Prime Minister, Robert Menzies.
</item 8>

<item 9>
Electoral district of Braidwood
Braidwood was an electoral district of the Legislative Assembly in the Australian state of New South Wales between 1859 and 1904, which included the town of
Braidwood.

 It replaced parts of the electoral district of United Counties of Murray and St Vincent and the electoral district of Southern Boroughs.

 In 1904 it was largely absorbed into the electoral district of Queanbeyan.
</item 9>

<item 10>
Beaconsfield, Victoria
Beaconsfield is a satellite suburb of Melbourne, Victoria, Australia, 46 km south-east of Melbourne's Central Business District, located within the Shire of
Cardinia local government area.

 Beaconsfield recorded a population of 6,714 at the 2016 Census.
</item 10>
=== END ITEM id=f803f756a5c873c0 ===

=== ITEM id=b075141ba5729be1 ===
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
Leah Flanagan
Leah Flanagan is a singer-songwriter from Sydney, NSW.

 Leah has appeared on Australian TV shows Spicks & Specks and RockQuiz, has released 2 albums and toured extensively through Australia with her music and part
of festival ensembles.

 Leah’s status as an Indigenous musician is one of pride and respect; she has helped make history with such recent essential albums as Archie Roach’s 25th
anniversary reissue of Charcoal Lane and Buried Country 1.5, The Story of Aboriginal Country Music.

 Her latest album Saudades was recorded at the studio of Midnight Oil's Jim Moginie and mixed by Paul McKercher (You Am I, Augie March)
</item 1>

<item 2>
Winwood (album)
Winwood is the first compilation album of music featuring Steve Winwood.

 This two-record set was issued in 1971 by United Artists Records and features music which Winwood performed with The Spencer Davis Group, Powerhouse, Traffic
and Blind Faith.

 UA Records issued this album after Winwood's band Traffic left UA when their home label Island Records set up their own American operation.

 Issued without Winwood's authorization as catalogue number UAS-9950, it was taken off the market after legal action by Winwood and Island Records.

 It was then reissued with minor changes as catalogue number UAS-9964.

 Currently out of print, it was issued on CD by Universal Music of Japan for the Japanese market.
</item 2>

<item 3>
Live at Winterland 1978 (Avengers album)
Live at Winterland 1978 is a live album by the Avengers.

 It was released through online music stores on February 16, 2010.

 The album features a recording of their set on January 14, 1978 at the Winterland Ballroom in San Francisco, CA, when they opened for the Sex Pistols on what
would become the Sex Pistols final show, before reuniting years later.

 The Sex Pistols' set has been released on an album of the same name.

 Songs from this album were previously released on a couple 7" bootlegs titled "Penelope" and "Summer of Hate".
</item 3>

<item 4>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 4>

<item 5>
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
</item 5>

<item 6>
Graph query row:
entities: ['Wangaratta Wahine', 'The Captain Matchbox Whoopee Band']
relationships: ['CREATED_BY']
</item 6>

<item 7>
Wangaratta Wahine
Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.

 It was recorded at top recording studio Armstrong Studios in Melbourne and the cover art was by famed Australian artist Michael Leunig.
</item 7>

<item 8>
He Will Have His Way
He Will Have His Way is a compilation album featuring male Australian and New Zealand musicians performing songs written by Neil Finn and Tim Finn (The Finn
Brothers), who are best known as members of Split Enz and Crowded House.

 It was released on 12 November 2010 and is a sequel to "She Will Have Her Way", a 2005 album featuring female Australian and New Zealand musicians.

 The two albums were later released together with extra tracks as the double CD "They Will Have Their Way".
</item 8>

<item 9>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 9>

<item 10>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 10>
=== END ITEM id=b075141ba5729be1 ===

=== ITEM id=efbfdb33c10dd858 ===
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
York
York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.

 The municipality is the traditional county town of the historic county of Yorkshire to which it gives its name.

 The city has a rich heritage and has provided the backdrop to major political events in England throughout much of its two millennia of existence.

 The city offers a wealth of historic attractions, of which York Minster is the most prominent, and a variety of cultural and sporting activities making it a
popular tourist destination for millions.
</item 1>

<item 2>
York Minster Police
York Minster Police is a small, specialised cathedral constabulary responsible for policing York Minster in York, United Kingdom.
</item 2>

<item 3>
Tripartite Indenture
The Tripartite Indenture was an agreement made in February 1405 between Owain Glyndŵr, Edmund Mortimer, and Henry Percy, 1st Earl of Northumberland, agreeing to
divide England and Wales up between them at the expense of Henry IV.

 Glyndŵr was to be given Wales, and a substantial part of the west of England, including the English portions of the Welsh Marches.

 Northumberland was to have received the north, as well as Northamptonshire, Norfolk, Warwickshire, and Leicestershire.

 The Mortimers were to have received the rest of southern England.
</item 3>

<item 4>
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
</item 6>

<item 7>
Earl Cowper
He represented Hertford in Parliament.

 He was succeeded by his eldest son, the aforementioned William Cowper, the third Baronet, who was elevated to the peerage as Baron Cowper in 1706 and made Earl
Cowper in 1718.

 In 1706 Lord Cowper married as his second wife Mary Clavering, daughter of John Clavering, of Chopwell, County Durham.
</item 7>

<item 8>
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
</item 8>

<item 9>
Copsi
Osulf, however, only ruled as earl until that autumn, when he was killed by an outlaw he was tracking.
</item 9>

<item 10>
York Minster
The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of
its kind in Northern Europe.

 The minster is the seat of the Archbishop of York, the second-highest office of the Church of England, and is the mother church for the Diocese of York and the
Province of York.

 It is run by a dean and chapter, under the Dean of York.

 The title "minster" is attributed to churches established in the Anglo-Saxon period as missionary teaching churches, and serves now as an honorific title.

 Services in the minster are sometimes regarded as on the High Church or Anglo-Catholic end of the Anglican continuum.
</item 10>
=== END ITEM id=efbfdb33c10dd858 ===

=== ITEM id=c2bf4b75f00092e2 ===
Gold sentences and facts:
Sentence 0: Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
  Fact 0.0: Giuseppe Fortunino Francesco Verdi was born on 9 or 10 October 1813.
  Fact 0.1: Giuseppe Fortunino Francesco Verdi died on 27 January 1901.
  Fact 0.2: Giuseppe Fortunino Francesco Verdi was an Italian opera composer.
Sentence 1: Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
  Fact 1.0: Charles Louis Ambroise Thomas was born on 5 August 1811.
  Fact 1.1: Charles Louis Ambroise Thomas died on 12 February 1896.
  Fact 1.2: Charles Louis Ambroise Thomas was a French composer.
  Fact 1.3: Charles Louis Ambroise Thomas is best known for his opera 'Mignon' (1866).
  Fact 1.4: Charles Louis Ambroise Thomas is best known for his opera 'Hamlet' (1868).
  Fact 1.5: The opera 'Hamlet' (1868) is after Shakespeare.
  Fact 1.6: Charles Louis Ambroise Thomas is best known as Director of the Conservatoire de Paris.
  Fact 1.7: Charles Louis Ambroise Thomas was Director of the Conservatoire de Paris from 1871 till his death.

Retrieved items:
<item 1>
Claudio Monteverdi
Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.

 A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance
and the Baroque periods of music history.
</item 1>

<item 2>
List of major opera composers
This list provides a guide to opera composers, as determined by their presence on a majority of compiled lists of significant opera composers.

 (See the "Lists Consulted" section for full details.)

 The composers run from Jacopo Peri, who wrote the first ever opera in late 16th century Italy, to John Adams, one of the leading figures in the contemporary
operatic world.

 The brief accompanying notes offer an explanation as to why each composer has been considered major.

 Also included is a section about major women opera composers, compiled from the same lists.

 For an introduction to operatic history, see opera.

 The organisation of the list is by birthdate.
</item 2>

<item 3>
Giuseppe Verdi (film)
Giuseppe Verdi is a 1938 Italian biographical film directed by Carmine Gallone and starring Fosco Giachetti, Gaby Morlay and Germana Paolieri.

 The film portrays the life of the composer Giuseppe Verdi (1813-1901).

 The casting of Giachetti as Verdi was intended to emphasise the composer's patriotism, as he had recently played patriotic roles in films such as "The White
Squadron".

 The film was made at the Cinecittà Studios in Rome.

 The film is also known by the alternative title The Life of Giuseppe Verdi.
</item 3>

<item 4>
Teatro Giuseppe Verdi
Teatro Giuseppe Verdi (the Giuseppe Verdi Theatre) is a small opera house located in a wing of the Rocca dei Marchesi Pallavicino on the Piazza Giuseppe Verdi
in Busseto, Italy, a town closely associated with the life of the opera composer, Giuseppe Verdi.

 From the 13th century, the “rocca” or “fortress” was the family’s palace; it is now the city hall after being acquired by the municipality in 1856.

 The theatre opened on 15 August 1868 and seats 300.
</item 4>

<item 5>
Ferdinando Provesi
Ferdinando Angelo Maria Provesi (1770 – 1833) was a native of Parma, Italy.

 He was regarded as one of the greatest Italian opera composers of the era .

 Provesi is best known as being an early tutor of Giuseppe Verdi when he was the "Maestro di cappella" (master of music) at the St. Bartolomeo cathedral in
Busseto (the town very close to Le Roncole, the village where Verdi was born.)

 Provesi was also director of the municipal music school and local Philharmonic Society.

 He began teaching Verdi in 1824 when the future composer was 11 years old.
</item 5>

<item 6>
The material surveys Italian opera history and related figures. Verdi’s Busseto theatre opened in 1868, seats 300, and occupies the former Pallavicino fortress.
Verdi studied under Ferdinando Provesi from age 11; French composer Ambroise Thomas wrote *Mignon* and *Hamlet*, whose 1868 premiere featured bass Armand
Castelmary, later famed for dying onstage in New York. Monteverdi, a Renaissance-to-Baroque pioneer,
</item 6>

<item 7>
Giuseppe Verdi
Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
</item 7>

<item 8>
Bernhard Bötel’s roles ranged from Mozart’s Belmonte to Verdi’s Duke of Mantua. The entries survey operas across eras: Gluck’s reform works *Orfeo ed Euridice*
(1762) and *Paride ed Elena* (1770); Hasse’s *Romolo ed Ersilia* (1765); Rossini’s *Mosè in Egitto* (1818); and later works by Donizetti, Bernstein,
</item 8>

<item 9>
The entries span opera, music, theatre, literature and Renaissance history. Bernhard Bötel sang roles from Mozart to Verdi; operas range from Gluck and Hasse to
Rossini and later composers. Commedia dell’arte’s masked stock characters, improvisation and lazzi are described. Musical highlights include Tchaikovsky’s
Second Piano Concerto, Debussy’s *Masques* and Bartók’s quartets. Other topics include Verdi’s Busseto theatre,
</item 9>

<item 10>
Ambroise Thomas
Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after
Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
</item 10>
=== END ITEM id=c2bf4b75f00092e2 ===

=== ITEM id=62cbbdbae99429d3 ===
Gold sentences and facts:
Sentence 0:  The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
  Fact 0.0: The breed is used for hunting in packs.
  Fact 0.1: The breed descends from the old Hound of Saintonge type of large hunting dog.
Sentence 1: The Kritikos Lagonikos (Cretan Hound) (Greek:Kρητικός Λαγωνικός) is a hunting breed of dog from the island of Crete, in Greece.
  Fact 1.0: The Kritikos Lagonikos, also called the Cretan Hound, is a hunting breed of dog.
  Fact 1.1: The Kritikos Lagonikos has the Greek name Κρητικός Λαγωνικός.
  Fact 1.2: The Kritikos Lagonikos is from the island of Crete, in Greece.

Retrieved items:
<item 1>
Grand Anglo-Français Blanc et Orange
The Grand Anglo-Français Blanc et Orange (FCI No.324) translated into English as the Great Anglo-French White and Orange Hound, is a breed of dog used in
hunting as a scenthound, usually in packs.

 It is one of the Anglo-French hound breeds which were created by crossing French scenthounds with English (Anglo) foxhounds.
</item 1>

<item 2>
Chien Français Blanc et Orange
The Chien Français Blanc et Orange (FCI No.316) is a breed of dog of the scenthound type, originating in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 2>

<item 3>
Petit Bleu de Gascogne
The Petit Bleu de Gascogne (FCI No.31) Is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog.

 The Petit Bleu de Gascogne is not a small (petite) dog, the name comes from its use on small game.
</item 3>

<item 4>
Braque Francais
The Braques français are hunting dogs, from a very old type of gun dog used for pointing the location of game birds for a hunter.

 There are two breeds of Braque français, both from the south of France, the Braque français, type Gascogne (French Pointing Dog - Gascogne type, larger size)
and the Braque français, type Pyrénées (French Pointing Dog - Pyrenean type, smaller size) They are popular hunting dogs in France, but are seldom seen
elsewhere.
</item 4>

<item 5>
Griffon Bleu de Gascogne
The Griffon Bleu de Gascogne (FCI No.32) Is a breed of dog of the scenthound type, originating in France, and is a versatile hunting dog, used on small and
large game, in packs or individually.

 The Griffon Bleu de Gascogne has a speckled, rough coat.
</item 5>

<item 6>
Chien Français Tricolore
The Chien Français Tricolore (FCI No.219) translated into English as the French Tricolour Hound, is a breed of dog of the scenthound type, originating in
France.

 The breed is used for hunting in packs.
</item 6>

<item 7>
Grand Bleu de Gascogne
The Grand Bleu de Gascogne (FCI No.22) is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog, and is an important breed in the ancestry of many other hounds.
</item 7>

<item 8>
The entries cover French hounds and regional wines. Gascony scenthounds—including Griffon, Petit and Grand Bleu, Basset Bleu, Ariegeois, and French and
Anglo-French breeds—are chiefly pack hunters; the Ariegeois is sociable, while “Petit” refers to small-game use, not size. Braque Français are southern French
pointing dogs in larger Gascogne and smaller Pyrénées types. Côtes de Gascogne is a
</item 8>

<item 9>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 9>

<item 10>
Grand Anglo-Français Blanc et Noir
The Grand Anglo-Français Blanc et Noir (FCI No.323) translated into English as the Great Anglo-French White and Black Hound, is a breed of dog used in hunting
as a scenthound, usually in packs.

 It is one of the Anglo-French hound breeds which were created by crossing French scenthounds with English (Anglo) Foxhounds.
</item 10>
=== END ITEM id=62cbbdbae99429d3 ===

=== ITEM id=15aa796f4399d920 ===
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
Spectrum (Say My Name)
"Spectrum" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 It was written by lead singer Florence Welch and Paul Epworth with production being handled by Epworth.

 The band premiered the song during a concert at Brooklyn's Creators Project on 15 October 2011, prior to the release of the album.

 The album version of "Spectrum" is a downtempo orchestral pop, R&B, krautrock and disco song.

 The accompanying music video for the song premiered on 30 May 2012.

 It was directed by David LaChapelle and John Byrne.
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
Lover to Lover
"Lover to Lover" is a song by English indie rock band Florence and the Machine.

 The song was released on 30 November 2012, as a digital download on iTunes as the fifth and final single from their second studio album "Ceremonials" (2011).

 The song was written by Florence Welch and Francis "Eg" White and produced by Paul Epworth.

 The Ceremonials Tour version of "Lover to Lover" is an edited version of the original with further production by Chris Hayden.

 The song became the band's first single to miss the top 200 in the UK Singles Chart.
</item 3>

<item 4>
Lungs (album)
Lungs is the debut studio album by English indie rock band Florence and the Machine, released on 3 July 2009 on Island Records.

 Frontwoman Florence Welch experimented with and honed the album's sound with bandmate Isabella Summers, while also collaborating with five record producers
including James Ford, Paul Epworth, Steve Mackey.

 The music on "Lungs" established Florence and the Machine as an internationally successful rock act, and also featured the group's successful mix of string
arrangements centered around Welch's vocal prowess.
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
What the Water Gave Me (painting)
What the Water Gave Me ("Lo que el agua me dio" in Spanish) is an oil painting by Frida Kahlo that was completed in 1938.

 It is sometimes referred to as "What I Saw in the Water".
</item 6>

<item 7>
Shake It Out
"Shake It Out" is a song by English indie rock band Florence and the Machine, released as the first official single from their second studio album,
"Ceremonials" (2011).

 It was written by Florence Welch and Paul Epworth, while production was handled by Epworth.

 The song was digitally released in Australia on 14 September 2011, and it was available in the United States on 19 October.

 It had its radio debut on XFM on 14 September 2011 in the United Kingdom.

 Welch revealed that the song was written within an hour and according to her it talked about shaking the regrets and the things that were haunting her.
</item 7>

<item 8>
What the Water Gave Me (song)
"What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.

 It was released on 23 August 2011 as the first promotional single from "Ceremonials".

 After the band performed the song in Berkeley, California on 12 June 2011, the studio version premiered on Florence and the Machine's official website.

 Welch decided to name the song "What the Water Gave Me" after seeing a Frida Kahlo painting of the same name.

 During an interview, she confirmed that the song was inspired by English writer Virginia Woolf.
</item 8>

<item 9>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 9>

<item 10>
Never Let Me Go (Florence and the Machine song)
"Never Let Me Go" is a song by English indie rock band Florence and the Machine from their second studio album, "Ceremonials" (2011).

 The song was written by Florence Welch and Paul Epworth while production was handled by Epworth.

 Island Records released the song as the third single from the album on 30 March 2012.

 The band first premiered the song during one of their concerts before the release of the album on 15 October 2011.

 A Clams Casino remix of the song was placed on the 12" white vinyl single which was available for pre-order on 22 March 2012.
</item 10>
=== END ITEM id=15aa796f4399d920 ===

=== ITEM id=6bd635779a50da47 ===
Gold sentences and facts:
Sentence 0: Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.
  Fact 0.0: Yucca is a genus of perennial shrubs and trees.
  Fact 0.1: Yucca is in the family Asparagaceae.
  Fact 0.2: Yucca is in the subfamily Agavoideae.
Sentence 1: Rodgersia is a genus of flowering plants in the Saxifragaceae family.
  Fact 1.0: Rodgersia is a genus of flowering plants.
  Fact 1.1: Rodgersia is in the Saxifragaceae family.
Sentence 2:  "Rodgersia" are herbaceous perennials originating from east Asia.
  Fact 2.0: Rodgersia are herbaceous perennials.
  Fact 2.1: Rodgersia originate from east Asia.

Retrieved items:
<item 1>
Rodgersia
Rodgersia is a genus of flowering plants in the Saxifragaceae family.

 "Rodgersia" are herbaceous perennials originating from east Asia.
</item 1>

<item 2>
Yucca aloifolia
Yucca aloifolia is the type species for the genus Yucca.

 Common names include aloe yucca, dagger plant, and Spanish bayonet.

 It grows in sandy soils, especially on sand dunes along the coast.

 "Yucca aloifolia" is native to the Atlantic and Gulf Coasts of the United States from southern Virginia south to Florida and west to the Texas Gulf Coast, to
Mexico along the Yucatán coast, and to Bermuda, and parts of the Caribbean.

 Normally Yucca aloifolia is grown in USDA zones 8 through 11.

 Yucca aloifolia is a popular landscape plant in beach areas along the lower East Coast from Virginia to Florida.
</item 2>

<item 3>
Yucca glauca
Yucca glauca (syn.

 "Yucca angustifolia") is a species of perennial evergreen plant, adapted to xeric (dry)growth conditions.

 It is also known as small soapweed, soapweed yucca, Spanish bayonet, Great Plains yucca and beargrass.
</item 3>

<item 4>
Hesperoyucca whipplei
Hesperoyucca whipplei (syn.

 "Yucca whipplei" ) (chaparral yucca, our Lord's candle, Spanish bayonet, Quixote yucca or foothill yucca is a species of flowering plant closely related to,
and formerly usually included in, the genus "Yucca".

 It is native to southern California, United States and Baja California, Mexico, where it occurs mainly in chaparral, coastal sage scrub, and oak woodland plant
communities at altitudes of 0–2500 m.
</item 4>

<item 5>
Yucca gloriosa var. tristis
Yucca gloriosa" var.

 "tristis (syn.

 Yucca recurvifolia, "Yucca gloriosa" var.

 "recurvifolia"), known as curve-leaf yucca, curved-leaved Spanish-dagger or pendulous yucca, is a variety of "Yucca gloriosa".

 It is often grown as an ornamental plant, but is native to the southeastern United States.

 In contrast to "Y. gloriosa" var.

 "tristis", the leaves of "Y. gloriosa" var.

 "gloriosa" are hard stiff, erect and narrower.
</item 5>

<item 6>
Yucca × schottii
Yucca" × "schottii is a plant species in the genus "Yucca", native to southern Arizona, southwestern New Mexico, and the northern parts of Sonora and Chihuahua.

.

 The common names are Schott's yucca, hoary yucca, and mountain yucca.

 The "×" in the name indicates that this is a nothospecies, regarded as being a natural hybrid between two other species.

 In this case, "Yucca" × "schottii" is believed to have originated as a hybrid between "Y. baccata" and "Y. madrensis".

 "Yucca" × "schottii" is firmly established and does reproduce freely in the wild.
</item 6>

<item 7>
Yucca brevifolia
Yucca brevifolia is a plant species belonging to the genus "Yucca".

 It is tree-like in habit, which is reflected in its common names: Joshua tree, yucca palm, tree yucca, and palm tree yucca.
</item 7>

<item 8>
Yucca
Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.

 Its 40-50 species are notable for their rosettes of evergreen, tough, sword-shaped leaves and large terminal panicles of white or whitish flowers.

 They are native to the hot and dry (arid) parts of the Americas and the Caribbean.

 Early reports of the species were confused with the cassava ("Manihot esculenta").

 Consequently, Linnaeus mistakenly derived the generic name from the Taíno word for the latter, "yuca" (spelled with a single "c").

 It is commonly found growing in rural graveyards and when in bloom the cluster of (usually pale) flowers on a thin stalk appear as floating apparitions.
</item 8>

<item 9>
Yucca rupicola
Yucca rupicola is a plant in the family Asparagaceae, known as the twistleaf yucca, twisted-leaf yucca, Texas yucca or twisted-leaf Spanish-dagger.

 The species was described by George Heinrich Adolf Scheele in 1850.

 This is a small, acaulescent plant with distinctive twisted leaves.

 It is native to the Edwards Plateau region of Texas and also to northeastern Mexico (Coahuila, Nuevo León).
</item 9>

<item 10>
Yucca jaliscensis
Yucca jaliscensis is a "Yucca" species native to the highlands of southwestern Mexico.

 Common names for this species include Jalisco Yucca, Jalisco Soapwort, Izote Yucca.

 It is native to mountainous areas at about 5000 feet in the States of Jalisco, Colima and Guanajuato.
</item 10>
=== END ITEM id=6bd635779a50da47 ===
