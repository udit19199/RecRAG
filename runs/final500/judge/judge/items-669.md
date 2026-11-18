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

This file has 6 items.

=== ITEM id=8c7a8b9ae42f7bdd ===
Gold sentences and facts:
Sentence 0: Urdu lyrics كشورِ حسين شاد باد ‬ پاک سرزمین شاد باد ‬ ارضِ پاکستان ! ‏ ‬ تُو نشانِ عزمِ عالی شان ‬ مرکزِ یقین شاد باد ‬ قوّتِ اُخوّتِ عوام ‬ پاک سرزمین کا نظام ‬ پائنده تابنده باد ! ‏ ‬ قوم ، ملک ، سلطنت ‬ شاد باد منزلِ مراد ‬ رہبرِ ترقّی و کمال ‬ پرچمِ ستاره و ہلال ‬ جانِ استقبال ! ‏ ‬ ترجمانِ ماضی ، شانِ حال ‬ سایۂ خدائے ذوالجلال ‬ Transliteration ( ALA - LC ) Translation ( poetical ) Translation ( literal ) Pāk sarzamīn shād bād Kishwar - i ḥasīn shād bād Tū nishān - i ʿazm - i ʿālī shān Arẓ - i Pākistān ! Markaz - i yaqīn shād bād May the holy land , stay glad ; Beautious realm , stay glad . Thou , the sign of high resolve -- O Land of Pakistan ! Citadel of faith , stay glad . Blessed be the sacred land , Happy be the bounteous realm . Thou symbol of high resolve , O Land of Pakistan ! Blessed be the citadel of faith . Pāk sarzamīn kā niz̤ām Quwwat - i Ukhuwwat - i ʿawām Qaum , mulk , salt̤anat Pāyindah tābindah bād ! Shād bād manzil - i murād Order of the holy land , Power of fraternity of the populace ; The nation , country , and domain ; Ever luminous remain ! The cherished goal , stay glad . The order of this sacred land , The might of the brotherhood of the people , May the nation , the country , and the state , Shine in glory everlasting ! Blessed be the goal of our ambition . Parcam - i sitārah o - hilāl Rahbar - i taraqqī o - kamāl Tarjumān - i māẓī , shān - i ḥāl Jān - i istiqbāl ! Sāyah - yi Khudā - yi Ẕū l - jalāl Flag with the star and crescent , The leader of progress and ascent , Dragoman of past , the pride of present ; Soul of the future ! Shadow of the God of grandeur . The flag of the crescent and star , Leads the way to progress and perfection , Interpreter of our past , glory of our present , inspiration for our future ! Shade of God , the Glorious and Mighty .
  Fact 0.0: The text gives Urdu lyrics about the Land of Pakistan.
  Fact 0.1: The text gives an ALA-LC transliteration of the Urdu lyrics.
  Fact 0.2: The text gives a poetical English translation of the lyrics.
  Fact 0.3: The text gives a literal English translation of the lyrics.
  Fact 0.4: The lyrics wish that the holy land stay glad.
  Fact 0.5: The lyrics wish that the beautiful realm stay glad.
  Fact 0.6: The lyrics describe the Land of Pakistan as the sign of high resolve.
  Fact 0.7: The lyrics describe the Land of Pakistan as the citadel of faith.
  Fact 0.8: The lyrics refer to the order of the holy land.
  Fact 0.9: The lyrics refer to the power of the brotherhood of the people.
  Fact 0.10: The lyrics wish that the nation, the country, and the state shine in everlasting glory.
  Fact 0.11: The lyrics wish that the cherished goal stay glad.
  Fact 0.12: The lyrics refer to the flag with the star and crescent.
  Fact 0.13: The lyrics describe the flag as the leader of progress and ascent.
  Fact 0.14: The lyrics describe the flag as an interpreter of the past.
  Fact 0.15: The lyrics describe the flag as the pride of the present.
  Fact 0.16: The lyrics describe the flag as the soul of the future.
  Fact 0.17: The lyrics describe the land as the shade of God, the Glorious and Mighty.

Retrieved items:
<item 1>
Qaumi Taranah
Music ( edit ) The Qaumi Taranah is a melodious and harmonious rendering of a three - stanza composition with a tune based on eastern music but arranged in such
a manner that it can be easily played by foreign bands . The music , composed by the Pakistani musician and composer , Ahmad G. Chagla in 1949 , reflects his
background in both eastern and western music . Twenty - one musical instruments and thirty - eight different tones are used to play the Qaumi Taranah , the
duration of which is 80 seconds . Lyrics ( edit ) The Urdu lyrics , written by the Pakistani Urdu - language poet , Hafeez Jullundhri in 1952 , have commonality
with Persian , making them understandable in both languages . No verse in the three stanza lyrics is repeated . The anthem has heavy Persian poetic vocabulary
and only uses one exclusively Urdu word ' kā ' . Urdu lyrics كشورِ حسين شاد باد ‬ پاک سرزمین شاد باد ‬ ارضِ پاکستان !
</item 1>

<item 2>
Qaumi Taranah
1954 -- Officially adopted as the national anthem and broadcast for the first time on Radio Pakistan on 13 August 1955 -- Sung by 11 famous Pakistani singers
including Ahmad Rushdi , Shamim Bano , Kokab Jehan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zwar Hussain , Akhtar Abbas , Ghulam Dastgir , Anwar Zaheer
and Akhtar Wassi 1996 -- Rendered in electric guitar for the first time by Pakistani rock band Junoon in their album Inquilaab 2009 -- Rendered as an acoustic
instrumental for the first time by Pakistani musician Jehangir Aziz Hayat 2011 -- On 14 August , 5,857 people gathered in a stadium in Karachi to sing the Qaumi
Taranah and set a new world record for most people gathered to sing a national anthem simultaneously . 2012 -- On 20 October , 42,813 people gathered in a
stadium in Lahore to sing the Qaumi Taranah and set a new world record for most people gathered to sing a national anthem simultaneously , which was certified
by Guinness World Records .
</item 2>

<item 3>
Qaumi Taranah
Archived from the original on 2007 - 10 - 26 . Retrieved 2007 - 12 - 09 . National Anthem of Pakistan Pakistan National Songs National symbols of Pakistan Main
symbols Qaumi Taranah ( national anthem ) State emblem of Pakistan Flag of Pakistan Iman , Ittihad , Nazm ( motto ) Lahore Resolution ( national document ) Urdu
( national language ) People Muhammad Ali Jinnah ( Father of the Nation ) Muhammad Iqbal ( National poet ) Fatima Jinnah ( Mother of the Nation ) Other symbols
Cedrus deodara ( national tree ) Chukar partridge ( national bird ) Jasminum officinale ( national flower ) Markhor ( national animal ) Structures Bab -
e-Khyber ( monument ) Faisal Mosque Mazar - e-Quaid ( mausoleum ) Minar - e-Pakistan ( monument ) Pakistan Monument ( monument ) Quaid - e-Azam Residency
National anthems of Asia National Abkhazia Afghanistan Armenia Artsakh ( Nagorno - Karabakh ) Azerbaijan Bahrain Bangladesh Bhutan Brunei Cambodia China Cyprus
East Timor Egypt Georgia India Indonesia Iran Iraq Israel Japan Jordan Kazakhstan North Korea South Korea
</item 3>

<item 4>
Qaumi Taranah
Official recognition to the national anthem , however , was not given until August 1954 . The NAC distributed records of the composed tune amongst prominent
poets , who responded by writing and submitting several hundred songs for evaluation by the NAC . Eventually , the lyrics written by Hafeez Jullundhri were
approved and the new national anthem was broadcast publicly for the first time on Radio Pakistan on 13 August 1954 , sung by Hafeez Jullundhri himself .
Official approval was announced by the Ministry of Information and Broadcasting on 16 August 1954 . The composer , Ahmed G. Chagla , died in 1953 , before the
new national anthem was officially adopted . In 1955 , there was a performance of the national anthem involving 11 major singers of Pakistan , including Ahmad
Rushdi , Kaukab Jahan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zawar Hussain , Akhtar Abbas , Ghulam Dastagir , Anwar Zaheer and Akhtar Wasi Ali .
</item 4>

<item 5>
Qaumi Taranah
Contents ( hide ) 1 History 2 Music 3 Lyrics 4 Milestones 5 See also 6 References 7 External links History ( edit ) In early 1948 , A.R. Ghani , a Muslim from
Transvaal , South Africa , offered two prizes of five thousand rupees each for the poet and composer of a new national anthem for the newly independent state of
Pakistan . The prizes were announced through a government press advertisement published in June 1948 . In December 1948 , the Government of Pakistan established
the National Anthem Committee ( NAC ) with the task of coming up with the composition and lyrics for the official national anthem of Pakistan . The NAC was
initially chaired by the Information Secretary , Sheikh Muhammad Ikram , and its members included several politicians , poets and musicians , including Abdur
Rab Nishtar , Ahmad G. Chagla and Hafeez Jullundhri . The NAC encountered early difficulties in finding suitable music and lyrics . When President Sukarno of
Indonesia became the first foreign head of state to visit Pakistan on 30 January 1950 , there was no Pakistani national anthem to be played .
</item 5>

<item 6>
Qaumi Taranah
http://mazhar.dk/film/singers/ahmadrushdi/ . Retrieved 2006 - 04 - 12 . Jump up ^ Mazhar Iqbal , Mazhar.dk . `` National Anthem of Pakistan '' . Retrieved 2006
- 04 - 12 . Jump up ^ Information Ministry , Government of Pakistan . `` Basic Facts '' . Retrieved 2006 - 04 - 12 . Jump up ^ ( 1 ) , National Anthem of
Pakistan Jump up ^ ( 2 ) , The National Anthem of Pakistan Jump up ^ ( 3 ) , Switch to Urdu from English as official language a hurdle for Pakistan Jump up ^ (
4 ) , History can not be falsely written and manufactured Jump up ^ ( 5 ) , Pakistan Basic Facts Jump up ^ Pasha , Muhammad A. English Composition ( Part II ) .
Lahore : Command Publications . Jump up ^ `` Pakistan creates new anthem record '' . The Express Tribune . 2011 - 08 - 12 . Retrieved 2012 - 01 - 01 . Jump up ^
`` Pakistan reclaims anthem singing record '' .
</item 6>

<item 7>
Qaumi Taranah
Sāyah - yi Khudā - yi Ẕū l - jalāl Flag with the star and crescent , The leader of progress and ascent , Dragoman of past , the pride of present ; Soul of the
future ! Shadow of the God of grandeur . The flag of the crescent and star , Leads the way to progress and perfection , Interpreter of our past , glory of our
present , inspiration for our future ! Shade of God , the Glorious and Mighty . Milestones ( edit ) 1947 -- The new state of Pakistan came into being on 14
August . 1949 -- Music for the Qaumi Taranah is composed by the Pakistani musical composer , Ahmad G. Chagla ( running time : 80 seconds ) . 1952 -- Verses
written by the Pakistani poet , Hafeez Jullundhri , are selected from amongst 723 entries .
</item 7>

<item 8>
Qaumi Taranah
BBC News . 2017 - 08 - 14 . Retrieved 2017 - 08 - 15 . Jump up ^ `` Indian A capella group sings Pakistan 's national anthem on 71st Independence Day '' .
Firstpost. 2017 - 08 - 14 . Retrieved 2017 - 08 - 15 . Jump up ^ Staff , Images ( 2017 - 08 - 12 ) . `` Indians sing the Pakistani national anthem as a gift for
our Independence Day '' . Images . Retrieved 2017 - 08 - 15 . External links ( edit ) This article contains Urdu text . Without proper rendering support , you
may see unjoined letters running left to right or other symbols instead of Urdu script . Wikimedia Commons has media related to : Qaumi Taranah ( category ) Pak
Sar Zameen Shad Bad Lyrics Pakistan : Qaumī Tarānah -- Audio of the national anthem of Pakistan , with information and lyrics Ministry of Information and
Broadcasting , Government of Pakistan . `` National Anthem '' . Archived from the original on 2007 - 10 - 26 .
</item 8>

<item 9>
Qaumi Taranah
Qaumi Taranah - wikipedia Qaumi Taranah Jump to : navigation , search Qaumi Taranah قومی ترانہ English : National Anthem of Pakistan National Anthem National
anthem of Pakistan Lyrics Hafeez Jullundhri , June 1952 Music Ahmad G. Chagla , 21 August 1949 Adopted 13 August 1954 Relinquished Present Audio sample Qaumi
Taranah ( Instrumental ) file help Ahmad G. Chagla composed the music of the National Anthem of Pakistan in 1949 Ahmed Rushdi recorded the National Anthem of
Pakistan in 1954 The Qaumi Taranah ( Urdu : قومی ترانہ ‬ ‎ , Qaumī Tarānah pronounced ( ˈqɔː. mi ˈt̪ə. rɑː. nɑ ) , lit .
</item 9>

<item 10>
Qaumi Taranah
rɑː. nɑ ) , lit . `` National Anthem '' ) , also known as Pāk Sarzamīn ( Urdu : پاک سرزمین ‬ ‎ , pronounced ( ˈpɑːk ˈsər. zə. miːn ) , lit . `` The Sacred Land
'' ) , is the national anthem of Pakistan . Its music was composed by Ahmad G. Chagla in 1949 , preceding the lyrics , which were written by Hafeez Jullundhri
in 1952 . It was officially adopted as Pakistan 's national anthem in August 1954 and was recorded in the same year by eleven major singers of Pakistan
including Ahmad Rushdi , Kaukab Jahan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zawar Hussain , Akhtar Abbas , Ghulam Dastagir , Anwar Zaheer and Akhtar
Wasi Ali .
</item 10>
=== END ITEM id=8c7a8b9ae42f7bdd ===

=== ITEM id=3ad16b0b45b61dba ===
Gold sentences and facts:
Sentence 0: Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double Down
  Fact 0.0: The title is Diary of a Wimpy Kid: The Getaway.
  Fact 0.1: The author is Jeff Kinney.
  Fact 0.2: The country is United States.
  Fact 0.3: The language is English.
  Fact 0.4: The series is Diary of a Wimpy Kid.
  Fact 0.5: The genres are children's novel and graphic novel.
  Fact 0.6: The publisher is Amulet Books (US).
  Fact 0.7: The publisher is Puffin Books (UK).
  Fact 0.8: The publication date is November 7, 2017 (worldwide).
  Fact 0.9: The media type is print (hardcover).
  Fact 0.10: The book has 217 story pages, 224 altogether.
  Fact 0.11: The ISBN is 978-1-4197-2545-6.
  Fact 0.12: It is preceded by Double Down.

Retrieved items:
<item 1>
Graph query row:
work: Diary of a Wimpy Kid: The Getaway
genre: Graphic Novel
</item 1>

<item 2>
Diary of a Wimpy Kid: The Getaway
Please discuss this issue on the article 's talk page . ( December 2017 ) This article needs more links to other articles to help integrate it into the
encyclopedia . Please help improve this article by adding links that are relevant to the context within the existing text . ( December 2017 ) ( Learn how and
when to remove this template message ) ( Learn how and when to remove this template message ) Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country
United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication
date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double
Down Diary of a Wimpy Kid : The Getaway is the twelfth book in the Diary of a Wimpy Kid series by Jeff Kinney .
</item 2>

<item 3>
Graph query row:
work: Diary of a Wimpy Kid: The Getaway
genre: Children's novel
</item 3>

<item 4>
Diary of a Wimpy Kid: The Getaway
He is even excited by the activities the resort offer which he sees on the bus . However , Frank and Susan are disappointed that the resort has since changed ,
and it is now split with a family side ( the Mild Side ) and a couple side ( the Wild Side ) . The family also discovers that they got the wrong luggage that is
locked , and their luggage was sent back home . They only have the clothes they put in their carry on luggage . They attempt to buy some new clothes , but Frank
wo n't buy them as they are too expensive . The family goes to explore the resort and end up at the pool . The Director of Fun shows up , however , and gets
everybody moving . Greg is forced to take part in two games . After this , Greg sees an iguana by the pool and is worried , so the family goes to get some lunch
. Greg and his family try to eat outside ; however , too many birds and bugs keep ruining their meals . Frank contracts a stomach bug from his water , so they
head back to the suite . The next day , Greg uses the shower which is outside .
</item 4>

<item 5>
Diary of a Wimpy Kid: The Getaway
Waterstones.com . Retrieved 10 December 2017 . ( hide ) The Diary of a Wimpy Kid series by Jeff Kinney Novels Diary of a Wimpy Kid Rodrick Rules The Last Straw
Dog Days The Ugly Truth Cabin Fever The Third Wheel Hard Luck The Long Haul Old School Double Down The Getaway Characters Greg Heffley Film series Diary of a
Wimpy Kid Rodrick Rules Dog Days The Long Haul Retrieved from `` https://en.wikipedia.org/w/index.
</item 5>

<item 6>
Diary of a Wimpy Kid: The Getaway
Diary of a Wimpy Kid : the Getaway - wikipedia Diary of a Wimpy Kid : the Getaway Jump to : navigation , search ( hide ) This article has multiple issues .
Please help improve it or discuss these issues on the talk page . ( Learn how and when to remove these template messages ) This article consists almost entirely
of a plot summary . It should be expanded to provide more balanced coverage that includes real - world context . Please edit the article to focus on discussing
the work rather than merely reiterating the plot . ( August 2016 ) ( Learn how and when to remove this template message ) This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
December 2017 ) ( Learn how and when to remove this template message ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page .
</item 6>

<item 7>
Diary of a Wimpy Kid: The Getaway
Release date ( edit ) The Getaway is the 12th book in the series , and was released everywhere except Canada and China on November 7 , 2017 . Reception ( edit )
A week since its release , it has received 4.5 / 5 stars on Waterstones with 786 reviews . References ( edit ) Jump up ^ Wimpy Kid ( 27 April 2017 ) . `` Wimpy
Kid LIVE Special 10th Anniversary Presentation and Cover Reveal of Book 12 ! '' . YouTube . Retrieved 10 December 2017 . Jump up ^ `` Cover Of New Diary Of A
Wimpy Kid Book Revealed By Jeff Kinney To Fans Worldwide At Live streamed Carnegie Hall Event Hosted By ABRAMS '' . Tristateupdate.com . Retrieved 10 December
2017 . Jump up ^ `` Diary of a Wimpy Kid : The Getaway by Jeff Kinney -- Waterstones '' . Waterstones.com . Retrieved 10 December 2017 .
</item 7>

<item 8>
Diary of a Wimpy Kid: The Getaway
They enter the line for security , but a child unclips a barrier separating the lines , and everybody gets scrambled . They go through security and are delayed
again by somebody who has to remove all of his piercings . Manny then gets his toy stuck in an escalator , delaying them more . As a result , the family panic ,
grab some luggage and dash to the gate . It turns out their flight is delayed . Greg goes to dry his sock but destroys it while doing so . He hears that the
flight is overbooked , and tries to stop his family boarding in exchange of a hotel stay and $300 ( soon $500 ) , but is stopped by Susan . A replacement crew
for the plane is required and they finally board . Greg is first impressed by the plane 's first class seats before he discovers that he has to sit in economy .
However , Frank collected enough air miles because of his business trips to sit in first class . In economy , Mom , Rodrick , and Manny are seated in the right
row . Greg finds out he has three seats to himself , until a couple with a baby arrives late on board .
</item 8>

<item 9>
Diary of a Wimpy Kid: The Getaway
The book was unveiled during the 2017 Diary of a Wimpy Kid Virtually Live Event which was live streamed via YouTube as part of the 10th anniversary of the first
book . The book was published and released on November 7 , 2017 . Contents ( hide ) 1 Plot 2 Release date 3 Reception 4 References Plot ( edit ) After many
frustrations , Greg Heffley 's parents decide to skip Christmas and travel by plane to Isla De Corales , the hotel they stayed at while on their honeymoon much
to Greg 's dismay . On Christmas Eve they drive to the airport , but are delayed by a series of mishaps . They encounter holiday traffic but arrive at the
airport . As they fail to unload in time at the drop off zone , Frank and Greg have to return later with the luggage . Greg steps in a slush puddle and they end
up missing the shuttle . In the airport , the family 's suitcases are too heavy , so they stuff some clothes into their carry on baggage .
</item 9>

<item 10>
Diary of a Wimpy Kid: The Getaway
The couple sit on either side of him with their baby . Greg becomes paranoid when he watches the plane 's safety video and begins to ask questions about
unfortunate events . Greg is very aggravated of the other passengers because most are n't watching the safety video . Mid flight , he feels boxed in when the
people in front recline and the person behind slides his feet on Greg 's armrest . Greg tries to recline his chair too but when he asks the flight attendant how
, he learns that he ca n't because the row behind him is an emergency exit . He tries to swap seats with Frank but Rodrick beats him to it . He stays in the
bathroom because the couple 's baby started to get annoying , but then he returns to his seat after experiencing turbulence . He somehow was able to sleep for
the rest of the flight . As they land on Christmas Day , Greg begins to understand why the family came there . They touch down and began to head to the resort ,
Isla de Corales . He is even excited by the activities the resort offer which he sees on the bus .
</item 10>
=== END ITEM id=3ad16b0b45b61dba ===

=== ITEM id=a1e5e47854b32c04 ===
Gold sentences and facts:
Sentence 0: Driver Seasons Entries Podiums Percentage Michael Schumacher 1991 -- 2006 , 2010 -- 2012 308 155 7001503200000000000 ♠ 50.32 % Lewis Hamilton 2007 -- 2018 211 119 7001564000000000000 ♠ 56.40 % Alain Prost 1980 -- 1991 , 1993 202 106 7001524800000099999 ♠ 52.48 % Sebastian Vettel 2007 -- 2018 202 101 7001500000000000000 ♠ 50.00 % 5 Fernando Alonso 2001 , 2003 -- 2018 296 97 7001327700000000000 ♠ 32.77 % 6 Kimi Räikkönen 2001 -- 2009 , 2012 -- 2018 276 93 7001337000000000000 ♠ 33.70 % 7 Ayrton Senna 1984 -- 1994 162 80 7001493800000000000 ♠ 49.38 % 8 Rubens Barrichello 1993 -- 2011 326 68 7001208600000000000 ♠ 20.86 % 9 David Coulthard 1994 -- 2008 247 62 7001251000000000000 ♠ 25.10 % 10 Nelson Piquet 1978 -- 1991 207 60 7001289900000000000 ♠ 28.99 % Source :
  Fact 0.0: The table lists drivers with their seasons, entries, podiums, and percentage.
  Fact 0.1: Michael Schumacher raced in the seasons 1991 to 2006 and 2010 to 2012.
  Fact 0.2: Michael Schumacher had 308 entries.
  Fact 0.3: Michael Schumacher had 155 podiums.
  Fact 0.4: Michael Schumacher has a percentage of 50.32%.
  Fact 0.5: Lewis Hamilton raced in the seasons 2007 to 2018.
  Fact 0.6: Lewis Hamilton had 211 entries.
  Fact 0.7: Lewis Hamilton had 119 podiums.
  Fact 0.8: Lewis Hamilton has a percentage of 56.40%.
  Fact 0.9: Alain Prost raced in the seasons 1980 to 1991 and 1993.
  Fact 0.10: Alain Prost had 202 entries.
  Fact 0.11: Alain Prost had 106 podiums.
  Fact 0.12: Alain Prost has a percentage of 52.48%.
  Fact 0.13: Sebastian Vettel raced in the seasons 2007 to 2018.
  Fact 0.14: Sebastian Vettel had 202 entries.
  Fact 0.15: Sebastian Vettel had 101 podiums.
  Fact 0.16: Sebastian Vettel has a percentage of 50.00%.
  Fact 0.17: Fernando Alonso is ranked 5th in the table.
  Fact 0.18: Fernando Alonso raced in the seasons 2001 and 2003 to 2018.
  Fact 0.19: Fernando Alonso had 296 entries.
  Fact 0.20: Fernando Alonso had 97 podiums.
  Fact 0.21: Fernando Alonso has a percentage of 32.77%.
  Fact 0.22: Kimi Räikkönen is ranked 6th in the table.
  Fact 0.23: Kimi Räikkönen raced in the seasons 2001 to 2009 and 2012 to 2018.
  Fact 0.24: Kimi Räikkönen had 276 entries.
  Fact 0.25: Kimi Räikkönen had 93 podiums.
  Fact 0.26: Kimi Räikkönen has a percentage of 33.70%.
  Fact 0.27: Ayrton Senna is ranked 7th in the table.
  Fact 0.28: Ayrton Senna raced in the seasons 1984 to 1994.
  Fact 0.29: Ayrton Senna had 162 entries.
  Fact 0.30: Ayrton Senna had 80 podiums.
  Fact 0.31: Ayrton Senna has a percentage of 49.38%.
  Fact 0.32: Rubens Barrichello is ranked 8th in the table.
  Fact 0.33: Rubens Barrichello raced in the seasons 1993 to 2011.
  Fact 0.34: Rubens Barrichello had 326 entries.
  Fact 0.35: Rubens Barrichello had 68 podiums.
  Fact 0.36: Rubens Barrichello has a percentage of 20.86%.
  Fact 0.37: David Coulthard is ranked 9th in the table.
  Fact 0.38: David Coulthard raced in the seasons 1994 to 2008.
  Fact 0.39: David Coulthard had 247 entries.
  Fact 0.40: David Coulthard had 62 podiums.
  Fact 0.41: David Coulthard has a percentage of 25.10%.
  Fact 0.42: Nelson Piquet is ranked 10th in the table.
  Fact 0.43: Nelson Piquet raced in the seasons 1978 to 1991.
  Fact 0.44: Nelson Piquet had 207 entries.
  Fact 0.45: Nelson Piquet had 60 podiums.
  Fact 0.46: Nelson Piquet has a percentage of 28.99%.
  Fact 0.47: The table ends with a 'Source' label that has no source given.

Retrieved items:
<item 1>
List of Formula One driver records
StatsF1.com . Retrieved 13 November 2016 . ^ Jump up to : `` Most wins & podiums for same team '' . StatsF1.com . Jump up ^ `` Most wins different constructors
'' . StatsF1.com . Retrieved 12 May 2015 . Jump up ^ `` Least wins in World Championship winning year '' . StatsF1.com . Retrieved 27 May 2015 . Jump up ^ ``
Most grand prix without podium '' . StatsF1.com . Retrieved 27 May 2015 . Jump up ^ `` Least career wins by World Champion '' . StatsF1.com . Retrieved 27 May
2015 . Jump up ^ `` The 2016 season in numbers '' . Formula1.com. 16 December 2016 . Retrieved 14 February 2018 . Jump up ^ `` Grands Prix before podium '' .
StatsF1.com . Retrieved 21 September 2017 . Jump up ^ `` Most 2nd places '' . StatsF1.com . Retrieved 26 August 2015 .
</item 1>

<item 2>
List of Formula One driver records
( edit ) ( only the first podium finish for each driver is listed ) Driver Age Place Race Max Verstappen 18 years , 228 days 1st 2016 Spanish Grand Prix Lance
Stroll 18 years , 239 days 3rd 2017 Azerbaijan Grand Prix Sebastian Vettel 21 years , 73 days 1st 2008 Italian Grand Prix Daniil Kvyat 21 years , 91 days 2nd
2015 Hungarian Grand Prix 5 Kevin Magnussen 21 years , 162 days 2nd 2014 Australian Grand Prix 6 Fernando Alonso 21 years , 237 days 3rd 2003 Malaysian Grand
Prix 7 Robert Kubica 21 years , 278 days 3rd 2006 Italian Grand Prix 8 Ralf Schumacher 21 years , 287 days 3rd 1997 Argentine Grand Prix 9 Elio de Angelis 21
years , 307 days 2nd 1980 Brazilian Grand Prix 10 Bruce McLaren 21 years ,
</item 2>

<item 3>
List of Formula One driver records
Pastor Maldonado ( 2012 Spanish Grand Prix ) , Mark Webber ( 2012 Monaco Grand Prix ) , Lewis Hamilton ( 2012 Canadian Grand Prix ) Fewest race wins in World
Championship winning year Mike Hawthorn ( 1958 ) Keke Rosberg ( 1982 ) World Champion with fewest career wins Mike Hawthorn , Phil Hill Most race wins in one
season without becoming World Champion 10 Lewis Hamilton ( 2016 ) Podiums Highest percentage of podium finishes in one season 100 Michael Schumacher , 17 podium
finishes out of 17 races in 2002 Most podiums for the same team 116 Michael Schumacher ( Ferrari ) Longest time between successive podium finishes 2,870 days
Alexander Wurz ( 1997 British Grand Prix -- 2005 San Marino Grand Prix ) Most consecutive podium finishes ( starting from debut ) 9 Lewis Hamilton ( 2007
Australian Grand Prix -- 2007 British Grand Prix ) Most podium finishes before a victory 15 Jean Alesi , Mika Häkkinen ,
</item 3>

<item 4>
List of Formula One driver records
1963 Dutch Grand Prix , 1963 French Grand Prix , 1963 Mexican Grand Prix , 1964 British Grand Prix , 1965 South African Grand Prix , 1965 French Grand Prix ,
1965 German Grand Prix Alberto Ascari 5 1952 French Grand Prix , 1952 German Grand Prix , 1952 Dutch Grand Prix , 1953 Argentine Grand Prix , 1953 British Grand
Prix Michael Schumacher 1994 Monaco Grand Prix , 1994 Canadian Grand Prix , 2002 Spanish Grand Prix , 2004 Australian Grand Prix , 2004 Hungarian Grand Prix
Lewis Hamilton 2014 Malaysian Grand Prix , 2015 Italian Grand Prix , 2017 Chinese Grand Prix , 2017 Canadian Grand Prix , 2017 British Grand Prix 5 Jackie
Stewart 1969 French Grand Prix , 1971 Monaco Grand Prix , 1971 French Grand Prix , 1972 United States Grand Prix Ayrton Senna 1985 Portuguese Grand Prix , 1989
Spanish Grand Prix , 1990 Monaco Grand Prix , 1990 Italian Grand Prix Nigel Mansell 1991 British Grand Prix , 1992 South African Grand Prix ,
</item 4>

<item 5>
List of Formula One driver records
Piquet 1987 Monaco -- 1987 Portuguese Michael Schumacher 2000 -- 2001 2000 Hungarian -- 2001 Brazilian 2004 European -- 2004 Italian Lewis Hamilton 2007 2007
Australian -- 2007 British Nico Rosberg 2015 2015 Australian -- 2015 British 2016 2016 Belgian -- 2016 Abu Dhabi Source : Most consecutive podium finishes from
first race of season ( edit ) Driver Season Consecutive podium finishes Podiums Michael Schumacher 2002 Australian -- Japanese 17 Fernando Alonso 2006 Bahrain
-- Canadian 9 Lewis Hamilton 2007 Australian -- British 2015 Australian -- British Sebastian Vettel 2011 Australian -- British Nico Rosberg 2015 Australian --
British 7 Nico Rosberg 2014 Australian -- Austrian 8 8 Niki Lauda 1976 Brazilian -- Swedish 7 Alain Prost 1988 Brazilian -- French Michael Schumacher 1994
Brazilian -- French Jenson Button 2009 Australian -- Turkish Source needed Youngest Drivers to score a podium finish ( edit ) ( only the first podium finish for
each driver is listed ) Driver Age Place Race Max
</item 5>

<item 6>
List of Formula One driver records
positions Youngest Oldest Most races without Fastest laps Total Percentage In one season Highest percentage in one season Youngest Oldest Most at the same GP
Podium finishes Total Percentage In one season Consecutive Consecutive from first race of season Youngest Oldest Most career starts without a podium Most career
podiums without a win Most career podiums without a World Championship Most races before scoring a podium finish Points Career points Total races finished in
the points Consecutive races finished in the points Highest average points per race started Highest percentage of races finished in the points Most in one
season Youngest Oldest Most points without a win Most without being World Champion World Champions with fewest career points Most races before scoring points
Most races without scoring points Race leaders Every lap , total For at least one lap , total Youngest Most laps Longest distance Most laps without a win Most
consecutive laps in the lead Multiple driver records Pole and win in same race Most wins from pole position in a season Pole , win and fastest lap in same race
Pole , win , fastest lap ,
</item 6>

<item 7>
List of Formula One driver records
ESPNF1 . Retrieved 9 January 2014 . Jump up ^ `` Most grand prix without pole '' . StatsF1.com . Retrieved 31 July 2017 . ^ Jump up to : `` Statistics Drivers
-- Fastests laps -- By age '' . StatsF1.com . Retrieved 2018 - 04 - 09 . ^ Jump up to : `` Podiums by number '' . StatsF1.com. 9 April 2017 . Retrieved 10 April
2017 . Jump up ^ `` Most consecutive podium finishes '' . StatsF1.com . Retrieved 10 May 2015 . Jump up ^ `` Grands Prix -- without podium '' . StatsF1.com. 23
July 2016 . Retrieved 23 July 2016 . Jump up ^ Collantine , Keith ( 11 April 2011 ) . `` Heidfeld sets record for most podiums without a win '' . F1 Fanatic .
Retrieved 10 January 2014 . Jump up ^ `` Top 100 F1 Drivers '' .
</item 7>

<item 8>
List of Formula One driver records
1953 Indianapolis 500 Tony Brooks 1957 British Grand Prix 9 4th race Bob Sweikert 1955 Indianapolis 500 Jacques Villeneuve 1996 European Grand Prix Source :
Most races before first win ( edit ) Start Driver Race 130th race Mark Webber 2009 German Grand Prix 123rd race Rubens Barrichello 2000 German Grand Prix 119th
race Jarno Trulli 2004 Monaco Grand Prix 113th race Jenson Button 2006 Hungarian Grand Prix 5 111th race Nico Rosberg 2012 Chinese Grand Prix 6 110th race
Giancarlo Fisichella 2003 Brazilian Grand Prix 7 96th race Mika Häkkinen 1997 European Grand Prix 8 95th race Thierry Boutsen 1989 Canadian Grand Prix 9 91st
race Jean Alesi 1995 Canadian Grand Prix 10 82nd race Eddie Irvine 1999 Australian Grand Prix Valtteri Bottas 2017 Russian Grand Prix Sources : Most races
without a win ( edit ) Driver Entries Starts Best result Andrea de Cesaris 214 208 2nd Nick
</item 8>

<item 9>
List of Formula One driver records
2004 -- 2017 269 11 41 7 Riccardo Patrese 1977 -- 1993 256 6 37 8 Jacques Laffite 1974 -- 1986 176 6 32 Jean Alesi 1989 -- 2001 201 10 Juan Pablo Montoya 2001
-- 2006 94 7 30 Source : Most races before scoring a podium finish ( edit ) Entry Driver Race 91st race Martin Brundle 1992 French Grand Prix 73rd race Mika
Salo 1999 German Grand Prix 68th race Jenson Button 2004 Malaysian Grand Prix 67th race Johnny Herbert 1995 Spanish Grand Prix Pedro de la Rosa 2006 Hungarian
Grand Prix 6 60th race Gianni Morbidelli 1995 Australian Grand Prix 7 57th race Felipe Massa 2006 European Grand Prix 8 56th race Mark Webber 2005 Monaco Grand
Prix 9 55th race Kamui Kobayashi 2012 Japanese Grand Prix Daniel Ricciardo 2014 Spanish Grand Prix Source : Points ( edit ) Throughout the history of the World
Championship ,
</item 9>

<item 10>
List of Formula One driver records
43 days 1959 United States Grand Prix 10 Jacques Laffite 41 years , 319 days 1985 European Grand Prix Source : Podium finishes ( edit ) Total podium finishes (
edit ) Driver Seasons Entries Podiums Percentage Michael Schumacher 1991 -- 2006 , 2010 -- 2012 308 155 7001503200000000000 ♠ 50.32 % Lewis Hamilton 2007 --
2018 211 119 7001564000000000000 ♠ 56.40 % Alain Prost 1980 -- 1991 , 1993 202 106 7001524800000099999 ♠ 52.48 % Sebastian Vettel 2007 -- 2018 202 101
7001500000000000000 ♠ 50.00 % 5 Fernando Alonso 2001 , 2003 -- 2018 296 97 7001327700000000000 ♠ 32.77 % 6 Kimi Räikkönen 2001 -- 2009 ,
</item 10>
=== END ITEM id=a1e5e47854b32c04 ===

=== ITEM id=b1de264a963849f0 ===
Gold sentences and facts:
Sentence 0: 2017 World Series Team ( Wins ) Manager ( s ) Season Houston Astros ( 4 ) A.J. Hinch 101 -- 61 (. 623 ) Los Angeles Dodgers ( 3 ) Dave Roberts 104 -- 58 (. 642 ) Dates October 24 -- November 1 MVP George Springer ( Houston ) Umpires Phil Cuzzi ( Games 1 -- 2 ) , Gerry Davis ( crew chief ) , Laz Díaz , Dan Iassogna , Bill Miller , Paul Nauert , Mark Wegner ( Games 3 -- 7 ) ALCS Houston Astros defeated New York Yankees , 4 -- 3 NLCS Los Angeles Dodgers defeated Chicago Cubs , 4 -- 1 Broadcast Television Fox Fox Deportes MLB International TV announcers Joe Buck , John Smoltz , Ken Rosenthal , and Tom Verducci ( Fox ) Rolando Nichols , Carlos Álvarez , Edgar Gonzalez , and Jaime Motta ( Fox Deportes ) Matt Vasgersian and Buck Martinez ( MLB International ) Radio ESPN Radio ESPN Deportes Radio Sportstalk 790 ( HOU -- English ) FM 94.1 HD3 ( HOU -- Spanish ) AM 570 LA Sports ( LAD -- English ) Univision America 1020 ( LAD -- Spanish ) Radio Korea 1540 ( LAD -- Korean ) Radio announcers Dan Shulman and Aaron Boone ( ESPN ) Eduardo Ortega , Renato Bermúdez , José Francisco Rivera , and Orlando Hernández ( ESPN Deportes ) Robert Ford and Steve Sparks ( HOU -- English ) Alex Treviño and Francisco Romero ( HOU -- Spanish ) Charley Steiner and Rick Monday ( LAD -- English ) Jaime Jarrín and Jorge Jarrín ( LAD -- Spanish ) Richard Choi and Chong Ho Yim ( LAD -- Korean ) World Series Program ← 2016 World Series
  Fact 0.0: The 2017 World Series table lists the Houston Astros as the team with 4 wins.
  Fact 0.1: The 2017 World Series table lists the Los Angeles Dodgers as the team with 3 wins.
  Fact 0.2: A.J. Hinch managed the Houston Astros in the 2017 season, with a record of 101 wins and 61 losses (.623).
  Fact 0.3: Dave Roberts managed the Los Angeles Dodgers in the 2017 season, with a record of 104 wins and 58 losses (.642).
  Fact 0.4: The 2017 World Series was played from October 24 to November 1.
  Fact 0.5: George Springer of the Houston Astros was the MVP of the 2017 World Series.
  Fact 0.6: Phil Cuzzi was an umpire for Games 1 and 2 of the 2017 World Series.
  Fact 0.7: Gerry Davis was the crew chief umpire of the 2017 World Series.
  Fact 0.8: Laz Diaz was an umpire of the 2017 World Series.
  Fact 0.9: Dan Iassogna was an umpire of the 2017 World Series.
  Fact 0.10: Bill Miller was an umpire of the 2017 World Series.
  Fact 0.11: Paul Nauert was an umpire of the 2017 World Series.
  Fact 0.12: Mark Wegner was an umpire for Games 3 to 7 of the 2017 World Series.
  Fact 0.13: The Houston Astros defeated the New York Yankees 4 to 3 in the 2017 ALCS.
  Fact 0.14: The Los Angeles Dodgers defeated the Chicago Cubs 4 to 1 in the 2017 NLCS.
  Fact 0.15: The 2017 World Series was broadcast on Fox television.
  Fact 0.16: The 2017 World Series was broadcast on Fox Deportes.
  Fact 0.17: The 2017 World Series was broadcast on MLB International TV.
  Fact 0.18: Joe Buck, John Smoltz, Ken Rosenthal and Tom Verducci were TV announcers on Fox.
  Fact 0.19: Rolando Nichols, Carlos Alvarez, Edgar Gonzalez and Jaime Motta were TV announcers on Fox Deportes.
  Fact 0.20: Matt Vasgersian and Buck Martinez were TV announcers on MLB International.
  Fact 0.21: The 2017 World Series was broadcast on ESPN Radio.
  Fact 0.22: The 2017 World Series was broadcast on ESPN Deportes Radio.
  Fact 0.23: Sportstalk 790 carried the Houston Astros English radio broadcast.
  Fact 0.24: FM 94.1 HD3 carried the Houston Astros Spanish radio broadcast.
  Fact 0.25: AM 570 LA Sports carried the Los Angeles Dodgers English radio broadcast.
  Fact 0.26: Univision America 1020 carried the Los Angeles Dodgers Spanish radio broadcast.
  Fact 0.27: Radio Korea 1540 carried the Los Angeles Dodgers Korean radio broadcast.
  Fact 0.28: Dan Shulman and Aaron Boone were radio announcers on ESPN.
  Fact 0.29: Eduardo Ortega, Renato Bermudez, Jose Francisco Rivera and Orlando Hernandez were radio announcers on ESPN Deportes.
  Fact 0.30: Robert Ford and Steve Sparks were English radio announcers for the Houston Astros.
  Fact 0.31: Alex Trevino and Francisco Romero were Spanish radio announcers for the Houston Astros.
  Fact 0.32: Charley Steiner and Rick Monday were English radio announcers for the Los Angeles Dodgers.
  Fact 0.33: Jaime Jarrin and Jorge Jarrin were Spanish radio announcers for the Los Angeles Dodgers.
  Fact 0.34: Richard Choi and Chong Ho Yim were Korean radio announcers for the Los Angeles Dodgers.
  Fact 0.35: The 2017 World Series followed the 2016 World Series.
Sentence 1: The 2017 World Series was the championship series of Major League Baseball 's ( MLB ) 2017 season . The 113th edition of the World Series , it was played between October 24 and November 1 . The series was a best - of - seven playoff between the National League ( NL ) champion Los Angeles Dodgers and the American League ( AL ) champion Houston Astros . It was sponsored by the internet television service YouTube TV and officially known as the World Series presented by YouTube TV .
  Fact 1.0: The 2017 World Series was the championship series of Major League Baseball's 2017 season.
  Fact 1.1: The 2017 World Series was the 113th edition of the World Series.
  Fact 1.2: The 2017 World Series was played between October 24 and November 1.
  Fact 1.3: The 2017 World Series was a best-of-seven playoff.
  Fact 1.4: The 2017 World Series was played between the National League champion Los Angeles Dodgers.
  Fact 1.5: The 2017 World Series was played between the American League champion Houston Astros.
  Fact 1.6: The 2017 World Series was sponsored by the internet television service YouTube TV.
  Fact 1.7: The 2017 World Series was officially known as the World Series presented by YouTube TV.

Retrieved items:
<item 1>
2005 World Series
1977 1978 1979 1980s -- 1990s 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000s -- 2010s 2000 2001 2002 2003 2005 2006
2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series champions Most Valuable Players Starting pitchers Babe Ruth
Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise ) Broadcasters TV ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7
1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series Category : World Series Chicago White Sox Based in Chicago ,
</item 1>

<item 2>
Game seven
1966 1967 1968 1969 1970 1971 1972 1973 1976 1977 1978 1979 1980s -- 1990s 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999
2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series champions Most
Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise ) Broadcasters TV
ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series Category : World
Series Stanley Cup Finals Stanley Cup awarded on challenge basis 1893 --
</item 2>

<item 3>
2017 World Series
Sports Business Journal . Retrieved October 4 , 2017 . External links ( edit ) 2017 World Series at Baseball Reference 2017 Major League Baseball postseason
Part of the 2017 Major League Baseball season World Series American League Championship Series National League Championship Series American League Division
Series National League Division Series American League Wild Card Game National League Wild Card Game American League teams Boston Red Sox Cleveland Indians
Houston Astros Minnesota Twins New York Yankees National League teams Arizona Diamondbacks Chicago Cubs Colorado Rockies Los Angeles Dodgers Washington
Nationals World Series No World Series was held in 1904 because the NL champions refused to participate ; no World Series was held in 1994 due to a players '
strike . 1900s -- 1910s 1900 1901 1902 1903 1904 1905 1906 1907 1908 1909 1910 1911 1912 1913 1914 1915 1916 1917 1918 1919 1920s -- 1930s 1920 1921 1922 1923
1924 1925 1926 1927 1928 1929
</item 3>

<item 4>
2017 World Series
The 113th edition of the World Series , it was played between October 24 and November 1 . The series was a best - of - seven playoff between the National League
( NL ) champion Los Angeles Dodgers and the American League ( AL ) champion Houston Astros . It was sponsored by the internet television service YouTube TV and
officially known as the World Series presented by YouTube TV . The Astros defeated the Dodgers , four games to three , to win their first World Series in
franchise history , and the first team from Texas to do so . Both teams set a World Series record with a combined total of 25 home runs throughout the entire
series , including a team record 15 home runs by the Astros , and hit a combined total of eight home runs in Game 2 to set the single game World Series mark .
Houston outfielder George Springer was named as the World Series Most Valuable Player ( MVP ) after hitting five home runs in the series to tie a World Series
record with Reggie Jackson in 1977 and Dodgers ' 2017 World Series representative Chase Utley in 2009 when he played for the Philadelphia Phillies .
</item 4>

<item 5>
2017 World Series
`` Overview of World Series home - field advantage '' . Major League Baseball . Retrieved November 2 , 2017 . We are just a week away from the World Series
presented by YouTube TV Jump up ^ `` Astros , Dodgers complete record - setting World Series '' . ESPN . November 2 , 2017 . Retrieved November 2 , 2017 . ^
Jump up to : Castrovince , Anthony ( November 1 , 2017 ) . `` Springer swats his way to MVP honors '' . Major League Baseball . Retrieved November 2 , 2017 .
Jump up ^ Schoenfield , David ( November 2 , 2017 ) . `` George Springer wins World Series MVP , ties record with five homers '' . ESPN . Retrieved November 2 ,
2017 . Jump up ^ Whicker , Mark ( October 22 , 2017 ) . `` Whicker : Dodgers , Houston Astros try to rekindle their 80s rivalry '' . Orange County Register .
Retrieved October 25 , 2017 .
</item 5>

<item 6>
Game seven
1 World Series 1.1. 2 League Championship Series 1.2 National Hockey League 1.3 National Basketball Association 1.4 Chinese Professional Baseball League 2
Comebacks 3 References Examples ( edit ) Major League Baseball ( edit ) In the MLB Postseason , game seven can occur in the League Championship Series and the
World Series ( the divisional series are best of five ) . In the World Series , there have been 39 decisive game sevens ( of the four World Series that were
contested on a best - of - nine basis , none went to a game nine , though all four included a non-decisive game seven ) . World Series ( edit ) World Series
that were decided on a game seven : All extra inning game sevens involve three franchises each with two appearances , the New York / San Francisco Giants (
losing both ) , the Washington Senators / Minnesota Twins ( winning both ) and the Cleveland Indians ( losing both ) . Home teams are 19 -- 20 in game sevens .
</item 6>

<item 7>
2017 World Series
1998 -- 2001 ) Jon Miller ( 1998 -- 2010 ) Dan Shulman ( 2002 -- present ) Gary Thorne ( 2008 -- 2009 ) Jon Sciambi ( 2010 -- present ) Analysts Kevin Kennedy (
1998 ) Dave Campbell ( 1999 -- 2010 ) Chris Singleton ( 2011 -- present ) Studio hosts Joe D'Ambrosio ( 1998 -- 2007 ) Marc Kestecher ( 2008 -- present ) AL
Championship Series 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 NL Championship Series 1998 1999 2000 2001
2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 AL Division Series 1998 1999 2000 2001 2002
</item 7>

<item 8>
2017 World Series
1997 1998 1999 2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series
champions Most Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise )
Broadcasters TV ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series
Category : World Series Fox Major League Baseball Related programs The Cheap Seats ( 2010 -- 2011 ) MLB Whiparound Major League Baseball Game of the Week
Thursday Night Baseball ( 1997 -- 2001 ) This Week in Baseball ( 2000 -- 2011 ) Related articles DirecTV N3D FoxBox FoxTrax Scooter Television contracts ( cable
) MLB Network World Series television ratings National coverage Fox ( 1996 -- present ) FS1 ( 2014 -- present
</item 8>

<item 9>
2017 World Series
2017 World Series - wikipedia 2017 World Series Jump to : navigation , search 2017 World Series Team ( Wins ) Manager ( s ) Season Houston Astros ( 4 ) A.J.
Hinch 101 -- 61 (. 623 ) Los Angeles Dodgers ( 3 ) Dave Roberts 104 -- 58 (. 642 ) Dates October 24 -- November 1 MVP George Springer ( Houston ) Umpires Phil
Cuzzi ( Games 1 -- 2 ) , Gerry Davis ( crew chief ) , Laz Díaz , Dan Iassogna , Bill Miller , Paul Nauert , Mark Wegner ( Games 3 -- 7 ) ALCS Houston Astros
defeated New York Yankees , 4 -- 3 NLCS Los Angeles Dodgers defeated Chicago Cubs , 4 -- 1 Broadcast Television Fox Fox Deportes MLB International TV announcers
Joe Buck , John Smoltz , Ken Rosenthal , and Tom Verducci ( Fox ) Rolando Nichols , Carlos Álvarez , Edgar Gonzalez ,
</item 9>

<item 10>
2017 World Series
and Tom Verducci ( Fox ) Rolando Nichols , Carlos Álvarez , Edgar Gonzalez , and Jaime Motta ( Fox Deportes ) Matt Vasgersian and Buck Martinez ( MLB
International ) Radio ESPN Radio ESPN Deportes Radio Sportstalk 790 ( HOU -- English ) FM 94.1 HD3 ( HOU -- Spanish ) AM 570 LA Sports ( LAD -- English )
Univision America 1020 ( LAD -- Spanish ) Radio Korea 1540 ( LAD -- Korean ) Radio announcers Dan Shulman and Aaron Boone ( ESPN ) Eduardo Ortega , Renato
Bermúdez , José Francisco Rivera , and Orlando Hernández ( ESPN Deportes ) Robert Ford and Steve Sparks ( HOU -- English ) Alex Treviño and Francisco Romero (
HOU -- Spanish ) Charley Steiner and Rick Monday ( LAD -- English ) Jaime Jarrín and Jorge Jarrín ( LAD -- Spanish ) Richard Choi and Chong Ho Yim ( LAD --
Korean ) World Series Program ← 2016 World Series The 2017 World Series was the championship series of Major League Baseball 's ( MLB ) 2017 season .
</item 10>
=== END ITEM id=b1de264a963849f0 ===

=== ITEM id=a35beb7a1c2e69e3 ===
Gold sentences and facts:
Sentence 0: The Super Bowl LI Halftime show took place on February 5 , 2017 , at NRG Stadium in Houston , Texas as part of Super Bowl LI . The show was headlined by Lady Gaga , who performed a medley of her songs , including newer material from her most recent studio album Joanne .
  Fact 0.0: The Super Bowl LI Halftime show took place on February 5, 2017.
  Fact 0.1: The Super Bowl LI Halftime show took place at NRG Stadium in Houston, Texas.
  Fact 0.2: The Super Bowl LI Halftime show took place as part of Super Bowl LI.
  Fact 0.3: The Super Bowl LI Halftime show was headlined by Lady Gaga.
  Fact 0.4: Lady Gaga performed a medley of her songs at the Super Bowl LI Halftime show.
  Fact 0.5: The medley included newer material from Lady Gaga's most recent studio album, Joanne.

Retrieved items:
<item 1>
Super Bowl LI halftime show
Retrieved October 1 , 2016 . Jump up ^ Halperin , Shirley ( September 29 , 2016 ) . `` It 's Official : Lady Gaga to Headline 2017 Super Bowl Halftime Show '' .
The Hollywood Reporter . Retrieved November 8 , 2016 . ^ Jump up to : Atkinson , Katie ( February 5 , 2017 ) . `` Super Bowl Halftime Will Be ' Uniquely Lady
Gaga ' : Pepsi Execs Promise ' One of the Greatest Halftime Shows Ever ' '' . Billboard . Archived from the original on February 21 , 2017 . Retrieved February
16 , 2017 . Jump up ^ Feeney , Nolan ( October 21 , 2016 ) . `` Lady Gaga : Joanne sealed my Super Bowl halftime show deal '' . Entertainment Weekly . Archived
from the original on November 23 , 2016 . Retrieved November 10 , 2016 . ^ Jump up to : Fisher , Kendall ( January 31 , 2017 ) .
</item 1>

<item 2>
Super Bowl LI halftime show
Live Design . Archived from the original on February 21 , 2017 . Retrieved February 20 , 2017 . Jump up ^ Orne , Neil ( February 6 , 2017 ) . `` Nashville
company played huge role in Super Bowl halftime show '' . WKRN - TV . Archived from the original on February 21 , 2017 . Retrieved February 20 , 2017 . Jump up
^ Sandberg , Marian ( February 5 , 2017 ) . `` Super Bowl LI Halftime Show 2017 Lighting Network System '' . Live Design . Archived from the original on
February 21 , 2017 . Retrieved February 20 , 2017 . ^ Jump up to : Lampert - Greaux , Ellen ( February 5 , 2017 ) . `` Bob Barnhart On The Super Bowl LI
Halftime Show '' . Live Design . Archived from the original on February 15 , 2017 . Retrieved February 20 , 2017 .
</item 2>

<item 3>
Super Bowl LI halftime show
Martens and Vera Wang made it a point . Multimedia ( edit ) During `` Million Reasons '' a field cast used Glow Motion LED Batons , creating a three -
dimensional moving pixels show . The production work on the show was done by designers Bruce Rodgers of Tribe Inc. and LeRoy Bennett , and lighting designer Bob
Barnhart of 22 Degrees . The stage was created by All Access Staging & Productions , who had also created previous Super Bowl halftime stages . Key features of
the stage consisted of a main platform , two towers , three star - shaped side stages , a piano dome and a stage lift . The towers were around 30 feet ( 9.1 m )
tall and could be carried by carts to the field for set - up . All Access had only 4.5 weeks to create the whole piece with the main stage itself , and required
17 trucks for storage and transport . The show utilized LED system from Verrents and sound by ATK Audio tek . Around 5,000 LED lights were embedded in the stage
, which was a record number . Special effects group Strictly FX was behind all the pyrotechnics .
</item 3>

<item 4>
Super Bowl LI halftime show
With 117.5 million television viewers , the Super Bowl LI halftime show was the second most - watched Super Bowl halftime show behind Katy Perry 's and had
higher average viewership than the game itself . Contents ( hide ) 1 Synopsis 2 Background 3 Development 3.1 Conception 3.2 Rehearsals 3.3 Fashion 3.4
Multimedia 3.5 Drones 4 Critical response 5 Commercial reception 6 Set list 7 Personnel 8 See also 9 References 10 External links Synopsis ( edit ) NRG Stadium
, where the Super Bowl LI and the halftime show took place The halftime show started on the roof of NRG Stadium . Gaga started singing `` God Bless America ''
as drones swarm behind her emitting red , blue and white color . She then sang Woody Guthrie 's `` This Land Is Your Land '' , followed by the ending of the
Pledge of Allegiance : `` One nation under God , indivisible , with liberty and justice for all '' . Gaga then dove from the rooftop as the drones behind formed
the design of the US flag .
</item 4>

<item 5>
Super Bowl LI halftime show
Barnhart recalled that `` ( the ) show was a little more complicated this year , due to the amount of scenery that we normally do n't have on a halftime show ''
, but it was better since there were no video backdrops . All around the roof of NRG Stadium , an array of 36 Clay Paky Sharpy fixtures were added , which lit
the Houston night skyline with light beams . Barnhart also arranged a backlight array fixture , using Philips Vari - Lite VL4000 and PRG Best Boy . LED tape was
used around the main production area . They were controlled by the lighting team and could light the stage from afar . Director Hamish Hamilton and his crew
devised their own way of capturing and transmitting the show for the live audience . Drones ( edit ) See also : Quadcopter § Art For performing on top of NRG
Stadium , Gaga was accompanied by a fleet of 300 synchronized , LED - equipped `` Shooting Star '' quadcopter drones provided by Intel , which formed an image
of the American flag behind the singer . It marked the first time that the drones had been used as part of a television broadcast .
</item 5>

<item 6>
Super Bowl LI halftime show
Super Bowl LI halftime show - wikipedia Super Bowl LI halftime show Jump to : navigation , search Super Bowl LI halftime show Date February 5 , 2017 Location
Houston , Texas Venue NRG Stadium Headliner Lady Gaga Sponsor Pepsi Director Hamish Hamilton Producer Ricky Kirshner Super Bowl halftime show chronology 50 (
2016 ) LI ( 2017 ) LII ( 2018 ) The Super Bowl LI Halftime show took place on February 5 , 2017 , at NRG Stadium in Houston , Texas as part of Super Bowl LI .
The show was headlined by Lady Gaga , who performed a medley of her songs , including newer material from her most recent studio album Joanne . Gaga 's
performance was critically acclaimed , with media outlets praising the show for demonstrating her artistic and musical qualities , and some arguing that it
ranked among the top Super Bowl halftime performances in history . The show was also noted for its acknowledgment of the LGBT community , the use of a swarm of
lighting - equipped drones for a pre-recorded intro scene , and its perceived political undertones .
</item 6>

<item 7>
Super Bowl LI halftime show
Jesus Arroyo , Arturo Martinez , Jeff Haas , Anthony Pozos , Micky Dymond , Staging Technician / Fabricator : Julio Roch LED Technicians : Zack Eastland , Blake
Chandler See also ( edit ) 2017 in American music 2017 in American television References ( edit ) Jump up ^ Nelson , Jeff ( October 4 , 2016 ) . `` Lady Gaga
Hopes to Deliver ' Powerful ' Super Bowl Halftime Show Like Greats Michael Jackson , Bruce Springsteen '' . People . Archived from the original on November 9 ,
2016 . Retrieved November 9 , 2016 . Jump up ^ Wooton , Dan ( August 8 , 2016 ) . `` Adele set to perform at the Super Bowl show as ' desperate ' chiefs bid to
land Brit star '' . The Sun . Archived from the original on November 9 , 2016 . Retrieved November 8 , 2016 . Jump up ^ `` Adele Reveals She Turned Down Super
Bowl 51 Halftime Gig '' . Rolling Stone . August 14 , 2016 .
</item 7>

<item 8>
Super Bowl LI halftime show
Gaga joined them wearing a football inspired costume and performed the track . As fireworks erupted heralding the end of the show , Gaga reached the edge of the
stage , shouted `` Super Bowl 51 '' , dropped her microphone , and jumped off stage catching a football . Background ( edit ) Gaga after performing the US
national anthem at the Super Bowl 50 on February 7 , 2016 Around August 2016 , media started reporting that singer Adele was in discussion with the National
Football League ( NFL ) to perform at the Super Bowl LI halftime show , to be held on February 5 , 2017 , at Houston 's NRG Stadium ; the show was sponsored by
Pepsi Zero Sugar . A representative from the NFL said that `` organisers are doing absolutely everything they can to try to persuade her to sign up '' . During
her Adele Live 2016 concert tour at Los Angeles , Adele clarified that although she had been offered to perform by the NFL , she had declined since she did not
believe herself to be suitable for the halftime show . The NFL also explained that they had not signed any musician till then and were still in discussion with
several artists .
</item 8>

<item 9>
Super Bowl LI halftime show
Billboard . Archived from the original on February 6 , 2017 . Retrieved February 5 , 2017 . Jump up ^ `` Super Bowl LI Halftime Show Credits And Staff '' . Live
Design . Archived from the original on February 21 , 2017 . Retrieved February 21 , 2017 . External links ( edit ) Wikimedia Commons has media related to Super
Bowl LI halftime show . Do n't Worry About What ' Joanne ' Songs Lady Gaga Is Gonna Perform at the Super Bowl Billboard Super Bowl LI Halftime Show 2017 Staging
Technical Drawings Live Design Super Bowl LI Halftime Show Set Drawings Live Design Will Lady Gaga 's Drones Win a Lion ?
</item 9>

<item 10>
Super Bowl LI halftime show
Toman also took into account the total time it usually takes for the show -- seven minutes for set up , 12 minutes for the main show , and another seven minutes
to dismantle -- and believed the singer could deliver `` power , punch , performance , singing , lights '' within that time frame . Pepsi sponsored the halftime
show for the fifth - consecutive year . Development ( edit ) Conception ( edit ) A long shot view of the stage during the performance of `` Just Dance '' in the
halftime show In October 2016 , Gaga told Entertainment Weekly she wanted to give `` the best show for the football fans , the ones that are watching at home ''
. Although indecisive about the set list , the singer wanted to play a mixture of her past and recent songs . She also confirmed that the NFL had heard Joanne
before its release , and decided to offer the halftime performance slot . Inspiration came from previous halftime show performances by Michael Jackson , Diana
Ross , and Bruce Springsteen , with Gaga remarking that she loved `` all the ( past shows ) '' .
</item 10>
=== END ITEM id=a35beb7a1c2e69e3 ===

=== ITEM id=59d3f36da11db8ef ===
Gold sentences and facts:
Sentence 0: City Image Country Population Year Mexico City Mexico 8,918,653 2015 New York City United States 8,550,405 2015 Los Angeles United States 3,971,883 2015 Toronto Canada 2,826,498 2015 Chicago United States 2,720,546 2015 6 Houston United States 2,296,224 2015 7 Havana Cuba 2,117,625 2015 8 Montreal Canada 1,753,034 2015 9 Ecatepec de Morelos Mexico 1,677,678 2015 10 Philadelphia United States 1,567,442 2015 11 Phoenix United States 1,563,025 2015 12 San Antonio United States 1,469,845 2015 13 Guadalajara Mexico 1,460,148 2015 14 Puebla Mexico 1,437,939 2015 15 San Diego United States 1,394,928 2015 16 Juárez Mexico 1,382,753 2015 17 León Mexico 1,349,224 2015 18 Dallas United States 1,300,092 2015 19 Tijuana Mexico 1,298,475 2015 20 Calgary Canada 1,230,915 2015 21 Tegucigalpa Honduras 1,190,230 2015 22 Zapopan Mexico 1,179,681 2015 23 Monterrey Mexico 1,109,171 2015 24 Managua Nicaragua 1,048,134 2015 25 Nezahualcóyotl Mexico 1,039,867 2015 26 San Jose United States 1,026,908 2015 27 Santo Domingo Dominican Republic 1,007,997 2015 28 Guatemala City Guatemala 994,078 2015 29 Port - au - Prince Haiti 987,310 2015 30 Naucalpan Mexico 970,012 2015 31 Ottawa Canada 956,710 2015 32 Austin United States 931,830 2015 33 Edmonton Canada 899,447 2016 34 Mérida Mexico 892,363 2015 35 Querétaro Mexico 878,931 2015 36 Toluca Mexico 873,536 2015 37 Jacksonville United States 868,031 2015 38 Chihuahua Mexico 867,736 2015 39 San Francisco United States 864,816 2015 40 Indianapolis United States 853,173 2015 41 Columbus United States 850,106 2015 42 Fort Worth United States 833,319 2015 43 Charlotte United States 827,097 2015 44 Hermosillo Mexico 819,999 2015 45 Saltillo Mexico 791,074 2015 46 Aguascalientes Mexico 785,942 2015 47 Mississauga Canada 761,300 2015 48 San Luis Potosí Mexico 759,854 2015 49 Veracruz Mexico 752,171 2015 50 San Pedro Sula Honduras 742,118 2015 51 Santiago de los Caballeros Dominican Republic 723,162 2015 52 Culiacán Mexico 720,798 2015 53 Winnipeg Canada 718,400 2015 54 Mexicali Mexico 717,628 2015 55 Cancún Mexico 711,682 2015 56 Acapulco Mexico 710,261 2015 57 Tlalnepantla Mexico 700,734 2015 58 Seattle United States 684,451 2015 59 Guadalupe Mexico 682,880 2015 60 Denver United States 682,545 2015 61 El Paso United States 681,124 2015 62 Chimalhuacán Mexico 679,811 2015 63 Detroit United States 677,116 2015 64 Washington , D.C. United States 672,228 2015 65 Kingston Jamaica 669,627 2015 66 Boston United States 667,137 2015 67 Tlaquepaque Mexico 664,193 2015 68 Memphis United States 655,770 2015 69 Nashville United States 654,610 2015 70 Torreón Mexico 653,150 2015 71 Vancouver Canada 648,608 2015 72 Portland United States 632,309 2015 73 Reynosa Mexico 631,611 2015 74 Oklahoma City United States 631,346 2015 75 Las Vegas United States 623,747 2015 76 Baltimore United States 621,849 2015 77 Brampton Canada 620,700 2016 78 Louisville United States 615,366 2015 79 Morelia Mexico 608,190 2015 80 Milwaukee United States 600,155 2015 81 Tuxtla Gutiérrez Mexico 598,710 2015 82 Apodaca Mexico 597,207 2015 83 Durango Mexico 590,893 2015 84 Villa Nueva Guatemala 564,686 2015 85 Albuquerque United States 559,121 2015 86 Hamilton Canada 556,359 2015 87 Quebec City Canada 540,994 2015 88 Tonalá Mexico 536,111 2015 89 Tucson United States 531,641 2015 90 Cuautitlán Izcalli Mexico 531,041 2015 91 Surrey Canada 526,004 2015 92 Ciudad López Mateos Mexico 523,296 2015 93 Tultitlán Mexico 520,557 2015 94 Fresno United States 520,052 2015 95 Carrefour Haiti 501,768 2015
  Fact 0.0: Mexico City is a city in Mexico with a population of 8,918,653 in 2015.
  Fact 0.1: New York City is a city in the United States with a population of 8,550,405 in 2015.
  Fact 0.2: Los Angeles is a city in the United States with a population of 3,971,883 in 2015.
  Fact 0.3: Toronto is a city in Canada with a population of 2,826,498 in 2015.
  Fact 0.4: Chicago is a city in the United States with a population of 2,720,546 in 2015.
  Fact 0.5: Houston is a city in the United States, ranked 6th, with a population of 2,296,224 in 2015.
  Fact 0.6: Havana is a city in Cuba, ranked 7th, with a population of 2,117,625 in 2015.
  Fact 0.7: Montreal is a city in Canada, ranked 8th, with a population of 1,753,034 in 2015.
  Fact 0.8: Ecatepec de Morelos is a city in Mexico, ranked 9th, with a population of 1,677,678 in 2015.
  Fact 0.9: Philadelphia is a city in the United States, ranked 10th, with a population of 1,567,442 in 2015.
  Fact 0.10: Phoenix is a city in the United States, ranked 11th, with a population of 1,563,025 in 2015.
  Fact 0.11: San Antonio is a city in the United States, ranked 12th, with a population of 1,469,845 in 2015.
  Fact 0.12: Guadalajara is a city in Mexico, ranked 13th, with a population of 1,460,148 in 2015.
  Fact 0.13: Puebla is a city in Mexico, ranked 14th, with a population of 1,437,939 in 2015.
  Fact 0.14: San Diego is a city in the United States, ranked 15th, with a population of 1,394,928 in 2015.
  Fact 0.15: Juárez is a city in Mexico, ranked 16th, with a population of 1,382,753 in 2015.
  Fact 0.16: León is a city in Mexico, ranked 17th, with a population of 1,349,224 in 2015.
  Fact 0.17: Dallas is a city in the United States, ranked 18th, with a population of 1,300,092 in 2015.
  Fact 0.18: Tijuana is a city in Mexico, ranked 19th, with a population of 1,298,475 in 2015.
  Fact 0.19: Calgary is a city in Canada, ranked 20th, with a population of 1,230,915 in 2015.
  Fact 0.20: Tegucigalpa is a city in Honduras, ranked 21st, with a population of 1,190,230 in 2015.
  Fact 0.21: Zapopan is a city in Mexico, ranked 22nd, with a population of 1,179,681 in 2015.
  Fact 0.22: Monterrey is a city in Mexico, ranked 23rd, with a population of 1,109,171 in 2015.
  Fact 0.23: Managua is a city in Nicaragua, ranked 24th, with a population of 1,048,134 in 2015.
  Fact 0.24: Nezahualcóyotl is a city in Mexico, ranked 25th, with a population of 1,039,867 in 2015.
  Fact 0.25: San Jose is a city in the United States, ranked 26th, with a population of 1,026,908 in 2015.
  Fact 0.26: Santo Domingo is a city in the Dominican Republic, ranked 27th, with a population of 1,007,997 in 2015.
  Fact 0.27: Guatemala City is a city in Guatemala, ranked 28th, with a population of 994,078 in 2015.
  Fact 0.28: Port-au-Prince is a city in Haiti, ranked 29th, with a population of 987,310 in 2015.
  Fact 0.29: Naucalpan is a city in Mexico, ranked 30th, with a population of 970,012 in 2015.
  Fact 0.30: Ottawa is a city in Canada, ranked 31st, with a population of 956,710 in 2015.
  Fact 0.31: Austin is a city in the United States, ranked 32nd, with a population of 931,830 in 2015.
  Fact 0.32: Edmonton is a city in Canada, ranked 33rd, with a population of 899,447 in 2016.
  Fact 0.33: Mérida is a city in Mexico, ranked 34th, with a population of 892,363 in 2015.
  Fact 0.34: Querétaro is a city in Mexico, ranked 35th, with a population of 878,931 in 2015.
  Fact 0.35: Toluca is a city in Mexico, ranked 36th, with a population of 873,536 in 2015.
  Fact 0.36: Jacksonville is a city in the United States, ranked 37th, with a population of 868,031 in 2015.
  Fact 0.37: Chihuahua is a city in Mexico, ranked 38th, with a population of 867,736 in 2015.
  Fact 0.38: San Francisco is a city in the United States, ranked 39th, with a population of 864,816 in 2015.
  Fact 0.39: Indianapolis is a city in the United States, ranked 40th, with a population of 853,173 in 2015.
  Fact 0.40: Columbus is a city in the United States, ranked 41st, with a population of 850,106 in 2015.
  Fact 0.41: Fort Worth is a city in the United States, ranked 42nd, with a population of 833,319 in 2015.
  Fact 0.42: Charlotte is a city in the United States, ranked 43rd, with a population of 827,097 in 2015.
  Fact 0.43: Hermosillo is a city in Mexico, ranked 44th, with a population of 819,999 in 2015.
  Fact 0.44: Saltillo is a city in Mexico, ranked 45th, with a population of 791,074 in 2015.
  Fact 0.45: Aguascalientes is a city in Mexico, ranked 46th, with a population of 785,942 in 2015.
  Fact 0.46: Mississauga is a city in Canada, ranked 47th, with a population of 761,300 in 2015.
  Fact 0.47: San Luis Potosí is a city in Mexico, ranked 48th, with a population of 759,854 in 2015.
  Fact 0.48: Veracruz is a city in Mexico, ranked 49th, with a population of 752,171 in 2015.
  Fact 0.49: San Pedro Sula is a city in Honduras, ranked 50th, with a population of 742,118 in 2015.
  Fact 0.50: Santiago de los Caballeros is a city in the Dominican Republic, ranked 51st, with a population of 723,162 in 2015.
  Fact 0.51: Culiacán is a city in Mexico, ranked 52nd, with a population of 720,798 in 2015.
  Fact 0.52: Winnipeg is a city in Canada, ranked 53rd, with a population of 718,400 in 2015.
  Fact 0.53: Mexicali is a city in Mexico, ranked 54th, with a population of 717,628 in 2015.
  Fact 0.54: Cancún is a city in Mexico, ranked 55th, with a population of 711,682 in 2015.
  Fact 0.55: Acapulco is a city in Mexico, ranked 56th, with a population of 710,261 in 2015.
  Fact 0.56: Tlalnepantla is a city in Mexico, ranked 57th, with a population of 700,734 in 2015.
  Fact 0.57: Seattle is a city in the United States, ranked 58th, with a population of 684,451 in 2015.
  Fact 0.58: Guadalupe is a city in Mexico, ranked 59th, with a population of 682,880 in 2015.
  Fact 0.59: Denver is a city in the United States, ranked 60th, with a population of 682,545 in 2015.
  Fact 0.60: El Paso is a city in the United States, ranked 61st, with a population of 681,124 in 2015.
  Fact 0.61: Chimalhuacán is a city in Mexico, ranked 62nd, with a population of 679,811 in 2015.
  Fact 0.62: Detroit is a city in the United States, ranked 63rd, with a population of 677,116 in 2015.
  Fact 0.63: Washington, D.C. is a city in the United States, ranked 64th, with a population of 672,228 in 2015.
  Fact 0.64: Kingston is a city in Jamaica, ranked 65th, with a population of 669,627 in 2015.
  Fact 0.65: Boston is a city in the United States, ranked 66th, with a population of 667,137 in 2015.
  Fact 0.66: Tlaquepaque is a city in Mexico, ranked 67th, with a population of 664,193 in 2015.
  Fact 0.67: Memphis is a city in the United States, ranked 68th, with a population of 655,770 in 2015.
  Fact 0.68: Nashville is a city in the United States, ranked 69th, with a population of 654,610 in 2015.
  Fact 0.69: Torreón is a city in Mexico, ranked 70th, with a population of 653,150 in 2015.
  Fact 0.70: Vancouver is a city in Canada, ranked 71st, with a population of 648,608 in 2015.
  Fact 0.71: Portland is a city in the United States, ranked 72nd, with a population of 632,309 in 2015.
  Fact 0.72: Reynosa is a city in Mexico, ranked 73rd, with a population of 631,611 in 2015.
  Fact 0.73: Oklahoma City is a city in the United States, ranked 74th, with a population of 631,346 in 2015.
  Fact 0.74: Las Vegas is a city in the United States, ranked 75th, with a population of 623,747 in 2015.
  Fact 0.75: Baltimore is a city in the United States, ranked 76th, with a population of 621,849 in 2015.
  Fact 0.76: Brampton is a city in Canada, ranked 77th, with a population of 620,700 in 2016.
  Fact 0.77: Louisville is a city in the United States, ranked 78th, with a population of 615,366 in 2015.
  Fact 0.78: Morelia is a city in Mexico, ranked 79th, with a population of 608,190 in 2015.
  Fact 0.79: Milwaukee is a city in the United States, ranked 80th, with a population of 600,155 in 2015.
  Fact 0.80: Tuxtla Gutiérrez is a city in Mexico, ranked 81st, with a population of 598,710 in 2015.
  Fact 0.81: Apodaca is a city in Mexico, ranked 82nd, with a population of 597,207 in 2015.
  Fact 0.82: Durango is a city in Mexico, ranked 83rd, with a population of 590,893 in 2015.
  Fact 0.83: Villa Nueva is a city in Guatemala, ranked 84th, with a population of 564,686 in 2015.
  Fact 0.84: Albuquerque is a city in the United States, ranked 85th, with a population of 559,121 in 2015.
  Fact 0.85: Hamilton is a city in Canada, ranked 86th, with a population of 556,359 in 2015.
  Fact 0.86: Quebec City is a city in Canada, ranked 87th, with a population of 540,994 in 2015.
  Fact 0.87: Tonalá is a city in Mexico, ranked 88th, with a population of 536,111 in 2015.
  Fact 0.88: Tucson is a city in the United States, ranked 89th, with a population of 531,641 in 2015.
  Fact 0.89: Cuautitlán Izcalli is a city in Mexico, ranked 90th, with a population of 531,041 in 2015.
  Fact 0.90: Surrey is a city in Canada, ranked 91st, with a population of 526,004 in 2015.
  Fact 0.91: Ciudad López Mateos is a city in Mexico, ranked 92nd, with a population of 523,296 in 2015.
  Fact 0.92: Tultitlán is a city in Mexico, ranked 93rd, with a population of 520,557 in 2015.
  Fact 0.93: Fresno is a city in the United States, ranked 94th, with a population of 520,052 in 2015.
  Fact 0.94: Carrefour is a city in Haiti, ranked 95th, with a population of 501,768 in 2015.

Retrieved items:
<item 1>
List of North American cities by population
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
List of North American cities by population
List of North American cities by Population - wikipedia List of North American cities by Population Jump to : navigation , search For the majority of cities in
North America , the most recent official population census results , estimates or short - term projections date to 2015 . This list compiles figures for all
North American cities with a population within city limits exceeding 500,000 that year . These figures do not reflect the population of the urban agglomeration
or metropolitan area which typically do not coincide with the administrative boundaries of the city . They refer to mid-2015 populations with the following
exceptions : Mexican cities , whose figures derive from the 2015 Intercensal Survey conducted by INEGI with a reference date of 15 March 2015 ; Calgary , whose
2015 municipal census had a reference date of April 1 ; Edmonton , which conducted municipal censuses in April 2014 and April 2016 but has not reported an
intercensal estimate of its 2015 population . The April 2016 census result is given instead .
</item 2>

<item 3>
List of North American cities by population
C. United States 672,228 2015 65 Kingston Jamaica 669,627 2015 66 Boston United States 667,137 2015 67 Tlaquepaque Mexico 664,193 2015 68 Memphis United States
655,770 2015 69 Nashville United States 654,610 2015 70 Torreón Mexico 653,150 2015 71 Vancouver Canada 648,608 2015 72 Portland United States 632,309 2015 73
Reynosa Mexico 631,611 2015 74 Oklahoma City United States 631,346 2015 75 Las Vegas United States 623,747 2015 76 Baltimore United States 621,849 2015 77
Brampton Canada 620,700 2016 78 Louisville United States 615,366 2015 79 Morelia Mexico 608,190 2015 80 Milwaukee United States 600,155 2015 81 Tuxtla Gutiérrez
Mexico 598,710 2015 82 Apodaca Mexico 597,207 2015 83 Durango Mexico 590,
</item 3>

<item 4>
List of North American cities by population
INEGI . p. 5 . Retrieved 2016 - 11 - 08 . ^ Jump up to : `` 2015 Municipal Affairs Population List '' ( PDF ) . Alberta Municipal Affairs . ISBN 978 - 1 - 4601
- 2630 - 1 . Retrieved July 17 , 2016 . ^ Jump up to : `` Encuesta Intercensal 2015 : Tabulados - descarga '' . INEGI . Archived from the original on 2016 - 11
- 13 . Retrieved 2016 - 11 - 08 . ^ Jump up to : Brinkhoff , Thomas . `` Mexico : Administrative Division '' . City Population . Retrieved 2016 - 11 - 08 . ^
Jump up to : `` Annual Estimates of the Resident Population for Incorporated Places of 50,000 or More , Ranked by July 1 , 2015 Population : April 1 , 2010 to
July 1 , 2015 '' . United States Census Bureau , Population Division . May 2016 .
</item 4>

<item 5>
List of North American cities by population
United States Census Bureau , Population Division . May 2016 . Retrieved 2016 - 05 - 19 . Jump up ^ `` Standard Geographical Classification ( SGC ) 2006 :
Economic Regions ( 3520 - Toronto , census subdivisions ) '' . Statistics Canada . 2014 - 04 - 01 . Archived from the original on 2015 - 05 - 30 . Retrieved
2015 - 06 - 11 . ^ Jump up to : `` Table 3.5 : Population and demographic factors of growth by census division , provinces and territories '' . Statistics
Canada . 2016 - 02 - 10 . Retrieved 2016 - 04 - 24 . Jump up ^ `` 3 . Población Proyectada al 30 de junio , por edades simples y años simples . AMBOS SEXOS .
Período 2015 - 2050 '' ( PDF ) . Proyecciones de la Población de Cuba 2015 - 2050 . Oficina Nacional de Estadísticas .
</item 5>

<item 6>
List of North American cities by population
300 2015 48 San Luis Potosí Mexico 759,854 2015 49 Veracruz Mexico 752,171 2015 50 San Pedro Sula Honduras 742,118 2015 51 Santiago de los Caballeros Dominican
Republic 723,162 2015 52 Culiacán Mexico 720,798 2015 53 Winnipeg Canada 718,400 2015 54 Mexicali Mexico 717,628 2015 55 Cancún Mexico 711,682 2015 56 Acapulco
Mexico 710,261 2015 57 Tlalnepantla Mexico 700,734 2015 58 Seattle United States 684,451 2015 59 Guadalupe Mexico 682,880 2015 60 Denver United States 682,545
2015 61 El Paso United States 681,124 2015 62 Chimalhuacán Mexico 679,811 2015 63 Detroit United States 677,116 2015 64 Washington , D.C. United States 672,228
2015 65 Kingston Jamaica 669,
</item 6>

<item 7>
List of North American cities by population
The April 2016 census result is given instead . Contents ( hide ) 1 List 2 See also 3 Notes 4 References List ( edit ) City Image Country Population Year Mexico
City Mexico 8,918,653 2015 New York City United States 8,550,405 2015 Los Angeles United States 3,971,883 2015 Toronto Canada 2,826,498 2015 Chicago United
States 2,720,546 2015 6 Houston United States 2,296,224 2015 7 Havana Cuba 2,117,625 2015 8 Montreal Canada 1,753,034 2015 9 Ecatepec de Morelos Mexico
1,677,678 2015 10 Philadelphia United States 1,567,442 2015 11 Phoenix United States 1,563,025 2015 12 San Antonio United States 1,469,845 2015 13 Guadalajara
Mexico 1,460,148 2015 14 Puebla Mexico 1,437,939 2015 15 San Diego United States 1,394,
</item 7>

<item 8>
List of North American cities by population
437,939 2015 15 San Diego United States 1,394,928 2015 16 Juárez Mexico 1,382,753 2015 17 León Mexico 1,349,224 2015 18 Dallas United States 1,300,092 2015 19
Tijuana Mexico 1,298,475 2015 20 Calgary Canada 1,230,915 2015 21 Tegucigalpa Honduras 1,190,230 2015 22 Zapopan Mexico 1,179,681 2015 23 Monterrey Mexico
1,109,171 2015 24 Managua Nicaragua 1,048,134 2015 25 Nezahualcóyotl Mexico 1,039,867 2015 26 San Jose United States 1,026,908 2015 27 Santo Domingo Dominican
Republic 1,007,997 2015 28 Guatemala City Guatemala 994,078 2015 29 Port - au - Prince Haiti 987,310 2015 30 Naucalpan Mexico 970,
</item 8>

<item 9>
List of North American cities by population
Retrieved 2016 - 05 - 01 . ( hide ) World 's largest cities City proper World ( Capitals ) Americas ( North Latin Central South ) Asia ( Arab world ASEAN Arabia
Far East ) Europe ( European Union Nordic ) Metropolitan area World Americas ( North South West Indies ) Europe ( European Union Nordic ) Asia Africa
Transborder Aggregate urban area † World Europe ( European Union Nordic ) Asia Urban agglomeration † North America Europe ( European Union ) Asia Historical
World Europe Related articles Global city Megacity Megalopolis Conurbation Suburbs † Aggregate urban areas are entire continuous built - up areas , grouping
together contiguous cities ( known as conurbations ) , whereas urban agglomerations are the built - up areas of individual cities . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=List_of_North_American_cities_by_population&oldid=822737139 '' Categories : Lists of cities by population Lists of
cities in the Americas North America - related lists Cities in North America Talk Contents About Wikipedia Español اردو 中文 Edit links This page was last edited
on 28 January 2018 , at 05 : 43 .
</item 9>

<item 10>
List of North American cities by population
310 2015 30 Naucalpan Mexico 970,012 2015 31 Ottawa Canada 956,710 2015 32 Austin United States 931,830 2015 33 Edmonton Canada 899,447 2016 34 Mérida Mexico
892,363 2015 35 Querétaro Mexico 878,931 2015 36 Toluca Mexico 873,536 2015 37 Jacksonville United States 868,031 2015 38 Chihuahua Mexico 867,736 2015 39 San
Francisco United States 864,816 2015 40 Indianapolis United States 853,173 2015 41 Columbus United States 850,106 2015 42 Fort Worth United States 833,319 2015
43 Charlotte United States 827,097 2015 44 Hermosillo Mexico 819,999 2015 45 Saltillo Mexico 791,074 2015 46 Aguascalientes Mexico 785,942 2015 47 Mississauga
Canada 761,300 2015 48 San Luis Potosí Mexico 759,
</item 10>
=== END ITEM id=59d3f36da11db8ef ===
