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

This file has 7 items.

=== ITEM id=936bb2ec844ec44b ===
Gold sentences and facts:
Sentence 0: The time that has elapsed since the Camp David Accords has left no doubt as to their enormous ramifications on Middle Eastern politics . Most notably , the perception of Egypt within the Arab world changed . With the most powerful of the Arab militaries and a history of leadership in the Arab world under Nasser , Egypt had more leverage than any of the other Arab states to advance Arab interests . Egypt was subsequently suspended from the Arab League from 1979 until 1989 .
  Fact 0.0: Time has elapsed since the Camp David Accords.
  Fact 0.1: The passage of time has left no doubt about the enormous ramifications of the Camp David Accords on Middle Eastern politics.
  Fact 0.2: The perception of Egypt within the Arab world changed.
  Fact 0.3: The change in perception of Egypt was the most notable effect.
  Fact 0.4: Egypt had the most powerful of the Arab militaries.
  Fact 0.5: Egypt had a history of leadership in the Arab world under Nasser.
  Fact 0.6: Egypt had more leverage than any other Arab state to advance Arab interests.
  Fact 0.7: Egypt was subsequently suspended from the Arab League.
  Fact 0.8: Egypt's suspension from the Arab League lasted from 1979 until 1989.

Retrieved items:
<item 1>
The Camp David Accords were two framework agreements signed by Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin on September 17, 1978,
after 12 days of secret negotiations at Camp David. Witnessed by U.S. President Jimmy Carter and signed at the White House, they included a framework that led
to the 1979 Egypt–Israel peace treaty.
</item 1>

<item 2>
The Camp David Accords (1978) were a major milestone in Arab–Israeli diplomacy, leading to the Egypt–Israel Peace Treaty of 1979. They included frameworks for
Middle East peace and an Egypt–Israel treaty, and are associated with U.S. President Jimmy Carter, Egyptian President Anwar Sadat, and Israeli Prime Minister
Menachem Begin. The material situates them among later peace efforts, including Madrid, Oslo, and the Israel–Jordan treaty, and lists historical references and
</item 2>

<item 3>
Camp David Accords
Most notably , the perception of Egypt within the Arab world changed . With the most powerful of the Arab militaries and a history of leadership in the Arab
world under Nasser , Egypt had more leverage than any of the other Arab states to advance Arab interests . Egypt was subsequently suspended from the Arab League
from 1979 until 1989 . When the Camp David accords were signed , Jordan 's King Hussein saw it as a slap to the face when Sadat volunteered Jordan 's
participation in deciding how functional autonomy would work . More specifically , Sadat effectively said that Jordan would have a role in how the West Bank
would be administered . Like the Rabat Summit Resolution , the Camp David Accords circumscribed Jordan 's objective to reassert its control over the West Bank .
Focusing as it did on Egypt , the Carter administration accepted Sadat 's claim that he could deliver Hussein . However , with Arab world opposition building
against Sadat , Jordan could not risk accepting the Accords without the support from powerful Arab neighbours , like Iraq , Saudi Arabia , and Syria . Hussein
consequently felt diplomatically snubbed .
</item 3>

<item 4>
Camp David Accords
Camp David Accords - wikipedia Camp David Accords Jump to : navigation , search Camp David Accords Framework for Peace in the Middle East and Framework for the
Conclusion of a Peace Treaty between Egypt and Israel Celebrating the signing of the Camp David Accords : Menachem Begin , Jimmy Carter , Anwar Sadat Type
Bilateral treaty Signed 17 September 1978 ( 1978 - 09 - 17 ) Location Washington , DC , United States Original signatories Egypt Israel Signatories Menachem
Begin ( Prime Minister of Israel ) Anwar Sadat ( President of Egypt ) Jimmy Carter ( President of the United States ) Ratifiers Egypt Israel Language English
French The Camp David Accords were signed by Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin on 17 September 1978 , following twelve
days of secret negotiations at Camp David . The two framework agreements were signed at the White House , and were witnessed by United States President Jimmy
Carter . The second of these frameworks ( A Framework for the Conclusion of a Peace Treaty between Egypt and Israel ) led directly to the 1979 Egypt -- Israel
Peace Treaty .
</item 4>

<item 5>
The Camp David Accords followed 14 months of diplomacy among Egypt, Israel, and the United States after Jimmy Carter became president. Carter’s goals included
Arab recognition of Israel, negotiated Israeli withdrawal from territories occupied in 1967 while ensuring Israel’s security, and an undivided Jerusalem. The
accords produced frameworks for Middle East peace and an Egyptian-Israeli treaty. Their Palestinian-territories framework was drafted without Palestinian
participation and condemned by the UN. Anwar Sadat and Menachem Begin
</item 5>

<item 6>
Camp David Accords
Lastly , the biggest consequence of all may be in the psychology of the participants of the Arab - Israeli conflict . The success of Begin , Sadat , and Carter
at Camp David demonstrated to other Arab states and entities that negotiations with Israel were possible -- that progress results only from sustained efforts at
communication and cooperation . Despite the disappointing conclusion of the 1993 Oslo Accords between the PLO and Israel , and even though the 1994 Israel --
Jordan peace treaty has not fully normalized relations with Israel , both of these significant developments had little chance of occurring without the precedent
set by Camp David . Public support Although most Israelis supported the Accords , the Israeli settler movement opposed them because Sadat would not agree to a
treaty in which Israel had any presence in the Sinai Peninsula at all , Israel had to withdraw from the entire Sinai Peninsula . Israeli settlers tried to
prevent the government from dismantling their settlements . In Israel , there is lasting support of the Camp David Peace Accords , which have become a national
consensus , supported by 85 % of Israelis according to a 2001 poll taken by the Jaffee Center for Strategic Studies ( Israel based ) .
</item 6>

<item 7>
Camp David Accords
Accompanied by their capable negotiating teams and with their respective interests in mind , both leaders converged on Camp David for 13 days of tense and
dramatic negotiations from 5 to 17 September 1978 . By all accounts , Carter 's relentless drive to achieve peace and his reluctance to allow the two men to
leave without reaching an agreement are what played the decisive role in the success of the talks . Carter 's advisers insisted on the establishment of an
Egyptian - Israeli agreement which would lead to an eventual solution to the Palestine issue . They believed in a short , loose , and overt linkage between the
two countries amplified by the establishment of a coherent basis for a settlement . However , Carter felt they were not `` aiming high enough '' and was
interested in the establishment of a written `` land for peace '' agreement with Israel returning the Sinai Peninsula and West Bank . Numerous times both the
Egyptian and Israeli leaders wanted to scrap negotiations , only to be lured back into the process by personal appeals from Carter .
</item 7>

<item 8>
Camp David Accords
The issues of Israeli settlement withdrawal from the Sinai and the status of the West Bank created what seemed to be an impasse . In response , Carter had the
choice of trying to salvage the agreement by conceding the issue of the West Bank to Begin , while advocating Sadat 's less controversial position on the
removal of all settlements from the Sinai Peninsula . Or he could have refused to continue the talks , reported the reasons for their failure , and allowed
Begin to bear the brunt of the blame . Carter chose to continue and for three more days negotiated . During this time , Carter even took the two leaders to the
nearby Gettysburg National Military Park in the hopes of using the American Civil War as a simile to their own struggle . Consequently , the 13 days marking the
Camp David Accords were considered a success . Partly due to Carter 's determination in obtaining an Israeli - Egyptian agreement , a full two - week pledge to
a singular international problem . Additionally , Carter was beneficiary to a fully pledged American foreign team . Likewise , the Israeli delegation had a
stable of excellent talent in Ministers Dayan and Weizman and legal experts Dr.
</item 8>

<item 9>
Camp David Accords
Meir Rosenne and Aharon Barak . Furthermore , the absence of the media contributed to the Accord 's successes : there were no possibilities provided to either
leader to reassure his political body or be driven to conclusions by members of his opposition . An eventual scrap of negotiations by either leader would have
proven disastrous , resulting in taking the blame for the summit 's failure as well as a disassociation from the White House . Ultimately , neither Begin nor
Sadat was willing to risk those eventualities . Both of them had invested enormous amounts of political capital and time to reach an agreement . Partial
agreements Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin acknowledge applause during a joint session of Congress in Washington , D.C.
, during which President Jimmy Carter announced the results of the Camp David Accords , 18 September 1978 . `` Remarks on the Signing of the Camp David Accords
'' Play media Jimmy Carter , seated with Egyptian President Anwar Sadat and Israeli Prime Minister Menachem Begin , makes statements at a Joint session of the
United States Congress following the Camp David Accords . Problems playing this file ? See media help .
</item 9>

<item 10>
The 1978 Camp David Accords comprised frameworks for Middle East peace and an Egypt–Israel treaty, which was signed in March 1979. Based on UN Security Council
Resolution 242, they proposed Palestinian self-government in the West Bank and Gaza, with “full autonomy” established within five years through negotiations
involving Israel, Egypt, Jordan, and Palestinian representatives. Israeli troop withdrawal would follow elections for a self-governing authority. The
framework’s terms were ambiguous; Jerusalem, Syria, Lebanon,
</item 10>
=== END ITEM id=936bb2ec844ec44b ===

=== ITEM id=49d4bd8ff0497eb1 ===
Gold sentences and facts:
Sentence 0: Dragon Ball Z ( ドラゴンボール ゼット , Doragon Bōru Zetto , commonly abbreviated as DBZ ) is the long - running anime sequel to the Dragon Ball TV series , adapted from the final twenty - six volumes of the Dragon Ball manga written by Akira Toriyama . The manga portion of the series debuted in Weekly Shōnen Jump in October 4 , 1988 and lasted until 1995 ; the anime adaptation premiered in Japan on Fuji Television on April 26 , 1989 , taking over its predecessor 's time slot , and ran until its end on January 31 , 1996 , lasting 291 episodes in Japan , and 276 episodes in the United States originally , although all 291 episodes were later broadcast when content from the first 67 episodes was restored .
  Fact 0.0: Dragon Ball Z (ドラゴンボール ゼット, Doragon Bōru Zetto, commonly abbreviated as DBZ) is the long-running anime sequel to the Dragon Ball TV series.
  Fact 0.1: Dragon Ball Z is adapted from the final twenty-six volumes of the Dragon Ball manga written by Akira Toriyama.
  Fact 0.2: The Dragon Ball Z manga portion debuted in Weekly Shōnen Jump in October 4, 1988.
  Fact 0.3: The Dragon Ball Z manga portion lasted until 1995.
  Fact 0.4: The Dragon Ball Z anime adaptation premiered in Japan on Fuji Television on April 26, 1989.
  Fact 0.5: The Dragon Ball Z anime adaptation took over its predecessor's time slot.
  Fact 0.6: The Dragon Ball Z anime adaptation ended on January 31, 1996.
  Fact 0.7: The Dragon Ball Z anime adaptation lasted 291 episodes in Japan.
  Fact 0.8: The Dragon Ball Z anime adaptation originally lasted 276 episodes in the United States.
  Fact 0.9: All 291 Dragon Ball Z episodes were later broadcast, after content from the first 67 episodes was restored.

Retrieved items:
<item 1>
List of Dragon Ball Z episodes
2001 91 -- 93 ( 76 -- 78 edited ) Super Saiyan Goku 26 000000002001 - 10 - 16 - 0000 October 16 , 2001 94 -- 96 ( 79 -- 81 edited ) Eleventh Hour 27
000000002001 - 11 - 13 - 0000 November 13 , 2001 97 -- 100 ( 82 -- 85 edited ) Fall of a Tyrant 28 000000002001 - 11 - 13 - 0000 November 13 , 2001 101 -- 104 (
86 -- 89 edited ) Namek 's End 29 000000002001 - 12 - 11 - 0000 December 11 , 2001 105 -- 107 ( 90 -- 92 edited ) Black Water Mist 30 000000002002 - 01 - 08 -
0000 January 8 , 2002 108 -- 110 ( 93 -- 95 edited ) Garlic Jr .
</item 1>

<item 2>
List of Dragon Ball Z episodes
2000 71 -- 74 ( 57 -- 60 edited ) The Summoning 20 000000002001 - 05 - 08 - 0000 May 8 , 2001 75 -- 77 ( 61 -- 63 edited ) Frieza Saga Transformation 21
000000002001 - 06 - 26 - 0000 June 26 , 2001 78 -- 81 ( 64 -- 66 edited ) Revealed 22 000000002001 - 07 - 10 - 0000 July 10 , 2001 82 -- 84 ( 67 -- 69 edited )
Death of a Prince 23 000000002001 - 08 - 21 - 0000 August 21 , 2001 85 -- 87 ( 70 -- 72 edited ) Clash 24 000000002001 - 08 - 14 - 0000 August 14 , 2001 88 --
90 ( 73 -- 75 edited ) Desperation 25 000000002001 - 09 - 04 - 0000 September 4 ,
</item 2>

<item 3>
List of Dragon Ball Z episodes
2007 6 75 -- 107 Frieza Saga Season Four February 19 , 2008 6 108 -- 139 Garlic Jr. / Trunks / Androids Saga Season Five May 27 , 2008 6 140 -- 165 Imperfect
Cell / Perfect Cell Saga Season Six September 16 , 2008 6 166 -- 194 Cell Games Saga Season Seven November 11 , 2008 6 195 -- 219 Great Saiyaman / World
Tournament Saga Season Eight February 10 , 2009 6 220 -- 253 Babidi / Majin Buu Saga Season Nine May 19 , 2009 6 254 -- 291 Fusion / Kid Buu Saga Funimation
Dragon Box Z Sets ( edit ) Name Date Discs Episodes Dragon Box Z , Vol. 1 November 17 , 2009 6 1 -- 42 Dragon Box Z , Vol. 2 February 16 , 2010 6 43 -- 84
Dragon Box Z , Vol. 3 May 4 , 2010 6 85 -- 126 Dragon Box Z , Vol.
</item 3>

<item 4>
List of Dragon Ball Z episodes
( Saiyan Sized Secret ) `` Ima da Gokū ! Subete o Kaketa Saigo no Ōwaza '' ( いまだ 悟空 ! すべて を 賭け た 最後 の 大 技 ) December 20 , 1989 May 3 , 1997 August 4 , 2005 (
re-dub ) 32 24 `` Battle Power Times Ten ! ! Vegeta 's Great Metamorphosis '' / `` Vegeta ... Saiyan Style ! '' ( Spirit Bomb Away ! ) `` Sentōryoku Jūbai ! !
Bejīta Daihenshin '' ( 戦闘 力 10 倍 ! ! ベジータ 大 変身 ) January 17 , 1990 May 10 , 1997 August 8 , 2005 ( re-dub ) 33 25 `` Do n't Die , Father ! ! This is the Depth
of Gohan 's Power '' / `` Stop Vegeta Now ! ''
</item 4>

<item 5>
List of Dragon Ball Z episodes
July 12 , 1989 October 25 , 1996 June 30 , 2005 ( re-dub ) 12 8 `` Catnap on the Serpentine Road Goku Takes a Tumble '' / `` Home For Infinite Losers '' (
Global Training ) `` Hebi no Michi de Inemuri Gokū ga Okkochiru '' ( 蛇 の 道 で い ねむり 悟空 が 落っこちる ) July 19 , 1989 November 1 , 1996 July 4 , 2005 ( re-dub ) 13 8
`` Hands Off ! Yama - sama 's Secret Fruit '' / `` Home For Infinite Losers '' ( Goz and Mez ) `` Te o Dasu na ! Enmasama no Himitsu no Kudamono '' ( 手 を 出す な !
</item 5>

<item 6>
List of Dragon Ball Z episodes
? '' ( 炎 の 決勝 ! ! 悟空 か パイクーハン か ! ? ) August 25 , 1993 September 13 , 2001 199 184 `` Do n't Let Victory Get Away ! ! Finish With an Ultra-Fast Kamehame - Ha ''
/ `` Goku vs. Pikkon '' `` Nigasu na Shōri ! ! Kimero Chōhaya Kamehameha '' ( 逃す な 勝利 ! ! 決めろ 超速 かめ はめ 波 ) September 1 , 1993 September 14 , 2001 200 185 ``
Seven Years Since Then ! Starting Today , I 'm a High School Student '' / `` Gohan Goes to High School '' `` Are kara Shichinen ! Kyō kara Boku wa Kōkōsei '' (
あれ から 7 年 !
</item 6>

<item 7>
List of Dragon Ball Z episodes
Season Episodes Japanese airdates English airdates Saiyan Saga 39 ( 1 -- 39 ) April 26 , 1989 -- March 7 , 1990 September 13 , 1996 -- September 20 , 1997 (
edited ) June 14 , 2005 -- August 18 , 2005 ( uncut ) Namek and Captain Ginyu Sagas 35 ( 40 -- 74 ) March 14 , 1990 -- January 16 , 1991 September 20 , 1997 --
April 4 , 1998 ( edited ) September 13 , 1999 -- September 20 , 1999 ( uncut ep. 68 -- 74 ) August 22 , 2005 -- October 10 , 2005 ( uncut ep. 40 -- 67 ) Frieza
Saga 33 ( 75 -- 107 ) January 23 , 1991 -- September 11 , 1991 September 21 , 1999 -- November 3 , 1999 Garlic Jr. , Trunks and Androids Sagas 32 ( 108 -- 139 )
September 18 ,
</item 7>

<item 8>
List of Dragon Ball Z episodes
The series was broadcast in eight separate near - continuous blocks ( 1 - 35 , 36 - 67 , 68 - 107 , 108 - 116 , 117 - 194 , 195 - 237 , 238 - 276 , and 277 -
291 ) , with breaks varying between four months to over a year between each block . In only one instance ( between episodes 194 and 195 ) was there actually
parity between the DVD release and the actual broadcast sequence in terms of the end of one `` season '' and the beginning of the next .
</item 8>

<item 9>
List of Dragon Ball Z episodes
! ブッ 飛べ ヤコン ) April 13 , 1994 October 18 , 2001 224 209 `` A Great Miscalculation ! ! Satan vs. Three Super-Warriors ! ? '' / `` Battle Supreme '' `` Daigosan !
! Satan tai Sannin no Chōsenshi ! ? '' ( 大 誤算 ! ! サタン VS3 人 の 超 戦士 ! ? ) April 20 , 1994 October 22 , 2001 225 210 `` Such Strong Kids ! ! No. 18 's Close Fight
! ? '' / `` Eighteen Unmasks '' `` Tsuyoi ze Chibikko ! ! Jūhachigō Daikusen ! ? '' ( 強い ぜ チビッコ ! ! 18 号 大 苦戦 ! ? )
</item 9>

<item 10>
List of Dragon Ball Z episodes
! ) March 25 , 1992 September 25 , 2000 134 119 `` Too Late to Do Anything ! ? The Ultimate Weapons to Kill Goku '' / `` Goku 's Assassin '' `` Subete ga
Teokure ka ! ? Gokū o Korosu Saishū Heiki '' ( すべて が 手遅れ か ! ? 悟空 を 殺す 最終 兵器 ) April 1 , 1992 September 26 , 2000 135 120 `` Good Looks and Super Power ! ? No
Blind Spot on No. 18 '' / `` Deadly Beauty '' `` Kawaii Kao de Chōpawā ! ? Jūhachigō ni Shikaku Nashi '' ( カワイイ 顔 で 超 パワー ! ? 18 号 に 死角 なし ) April 15 , 1992
September 27 , 2000 136 121 `` Nobody Is Able to Stop Them ... Is This the End of the Z Warriors !
</item 10>
=== END ITEM id=49d4bd8ff0497eb1 ===

=== ITEM id=4df16687308afda8 ===
Gold sentences and facts:
Sentence 0: Jane and Rafael start saving up to buy a one - bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo , the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine weeks , and Donna dies of unforeseen consequences . Xiomara is shown not to be doing well due to the chemotherapy , Alba and Rafael are still in a fight and River and Rogelio are still arguing over the Passions of Santos adaptation . Alba breaks down to Jane , and she and Rafael make up . Xiomara bonds with River over telenovelas , after she accidentally eats a pot brownie , and she agrees to do the Passions of Steve and Brenda the Latin American way . Jane Ramos gets herself disbarred , and as a result gets the case against Petra thrown out .
  Fact 0.0: Jane and Rafael start saving up to buy a one-bedroom apartment.
  Fact 0.1: Jane and Rafael's plan to buy the apartment falls apart when Mateo breaks an antique $4,000 clock.
  Fact 0.2: Rafael moves into the Villanueva house.
  Fact 0.3: After Alba hits Mateo, Alba and Rafael get into a heated argument.
  Fact 0.4: Rafael moves back out of the Villanueva house after the argument.
  Fact 0.5: Xiomara starts chemotherapy.
  Fact 0.6: Xiomara makes a friend named Donna.
  Fact 0.7: The timeline moves forward nine weeks.
  Fact 0.8: Donna dies of unforeseen consequences.
  Fact 0.9: Xiomara is shown not to be doing well due to the chemotherapy.
  Fact 0.10: Alba and Rafael are still in a fight.
  Fact 0.11: River and Rogelio are still arguing over the Passions of Santos adaptation.
  Fact 0.12: Alba breaks down to Jane.
  Fact 0.13: Alba and Rafael make up.
  Fact 0.14: Xiomara bonds with River over telenovelas.
  Fact 0.15: Xiomara accidentally eats a pot brownie.
  Fact 0.16: Xiomara agrees to do the Passions of Steve and Brenda the Latin American way.
  Fact 0.17: Jane Ramos gets herself disbarred.
  Fact 0.18: As a result of Jane Ramos being disbarred, the case against Petra is thrown out.
Sentence 1: No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 65 `` Chapter Sixty - Five '' Brad Silberling Jennie Snyder Urman & Paul Sciarrotta October 13 , 2017 ( 2017 - 10 - 13 ) 0.68 Picking up immediately after where Season 3 left off , Jane is reunited with her first love , Adam , who found the love letter that Michael wrote . She is surprised , but happy , to see him . However , both Xo and Alba are less thrilled that he is back . Meanwhile , Rafael loses ownership of the hotel to his sister and as a result , he finds himself living at Jane 's house , broke and homeless . Jane and Rafael learn that the missing Petra has been found after she got away from Anezka and they rush to her side , but Anezka is nowhere to be found . When Petra returns to the hotel , Luisa thinks that Petra is Anezka . Petra goes along with the game , until the real Anezka returns . Elsewhere , Xo and Rogelio 's married life is interrupted when they confront the realization that Darci is having Rogelio 's baby and she wants full custody . 66 `` Chapter Sixty - Six '' Gina Lamar Valentina Garza & Jessica O'Toole & Amy Rardin October 20 , 2017 ( 2017 - 10 - 20 ) 0.61 Jane 's relationship with Adam helps bring out her `` fun side '' and reminds her that she is n't just a mom . However , when Adam gets a job opportunity out of state , he asks for Jane 's opinion on if he should stay or go . Meanwhile , Rogelio continues to face issues with his co-star at work and things are about to get even worse . Xo is helping to take care of Darci to prove that Rogelio is committed to being a father to the baby . Also , Petra and Rafael begin working together to try and buy back the hotel from Luisa , but they are running into trouble involving Anezka , causing tension between them . 67 `` Chapter Sixty - Seven '' Fernando Sariñana Carolina Rivera & Micah Schraft October 27 , 2017 ( 2017 - 10 - 27 ) 0.60 Jane and Adam 's relationship seems to be moving on , and Jane thinks that it might be time for him to meet her son Mateo , but Rafael is not happy with this idea . Meanwhile , as the arrival of Darci 's baby approaches , Rogelio 's jealously of her new boyfriend increases as well . Trying to be supportive , Xo encourages Rogelio to step up and be the father he was meant to be . Elsewhere , Rafael is determined to get his hotel and life back from Luisa and will do whatever he has to do to make it happen , with or without Petra 's help . 68 `` Chapter Sixty - Eight '' Gina Lamar Deirdre Shaw & Chantelle M. Wells November 3 , 2017 ( 2017 - 11 - 03 ) 0.69 Jane becomes angry when she learns that Rafael introduced Mateo to his new girlfriend , Katherine , without her consent because she thinks Katherine is a negative influence on both Rafael and Mateo . Jane turns to Petra for help with the situation . Meanwhile , Rogelio and Darci struggle to choose a name for the baby . Elsewhere , Xo , who is focused on opening her own dance studio , decides to go into business with an unlikely foe . 69 5 `` Chapter Sixty - Nine '' Stuart Gillard Paul Sciarrotta November 10 , 2017 ( 2017 - 11 - 10 ) 0.65 After Adam reveals his dating past to Jane , she becomes so unsure and insecure that it puts a strain on their relationship . Meanwhile , Lina seeks Jane 's advice about whether she should get married or not . Petra hears some important information that could help Rafael get the hotel back . Also , Xo asks Rogelio to consider having a medical procedure ( vasectomy ) done , but he is very hesitant . 70 6 `` Chapter Seventy '' Melanie Mayron Carolina Rivera & Micah Schraft November 17 , 2017 ( 2017 - 11 - 17 ) 0.61 Jane 's book is finally being released but her plans to have a party at the local bookshop where she was inspired to be a writer hit a snag . Always wanting to help Jane , Rogelio devises a plan to save the day , but one of his arch - rivals threatens to ruin the day . Meanwhile , Rafael gets a new job , but it is not exactly what he was hoping for . Also , Petra has come up with an interesting strategy to get the Marbella Hotel back from Luisa , but Rafael wants nothing to do with the scheme . 71 7 `` Chapter Seventy - One '' Micah Schraft Valentina Garza & Deidre Shaw December 8 , 2017 ( 2017 - 12 - 08 ) 0.65 Anezka is revealed to have faked her death to see where her loyalties lie ; initially , she trusts Petra after believing she saw her cry , but after finding out she was n't crying because of her fake death , she threatens Petra . Later , Anezka falls from Petra 's apartment to her death ... for real . Meanwhile , Xiomara and Jane set off on Jane 's book tour , when Alba decides to tag along . After they get back , Jane and Rafael share a kiss . 72 8 `` Chapter Seventy - Two '' Melanie Mayron Carolina Rivera & Paul Sciarrotta January 26 , 2018 ( 2018 - 01 - 26 ) 0.68 Mateo starts kindergarten and invites a friend over to Xiomara and Rogelio 's house . Jane and Rafael try to cover up that they are pretending to live in their house so they do n't get kicked out . Jane and Rafael show mixed emotions following the kiss and try again , which Mateo notices . He then proclaims : `` Does this mean you 're going to get married ? '' After Xo convinces Rogelio to attend couples therapy , they both are surprised by their emotions . Meanwhile , Petra hires a lawyer named Jane Ramos ( Rosario Dawson ) , to ensure she gets no charges pending Anezka 's death . Also , Rafael moves out of Jane 's house into his new apartment . 73 9 `` Chapter Seventy - Three '' Eric Lea Valentina Garza & Chantelle M. Wells February 2 , 2018 ( 2018 - 02 - 02 ) 0.67 When Jane and Rafael finally realize that the uncertain state of their relationship ( or lack of it ) is stressing out Mateo , they decide that they need to make a decision . Meanwhile , Petra is on the outs with Jane and Rafael after the Luisa fiasco , so she tries to figure out a way to get back into their good graces . Elsewhere , Rogelio and Xo are both contemplating their career choices . 74 10 `` Chapter Seventy - Four '' Gina Rodriguez Micah Schraft & Paul Sciarrotta February 9 , 2018 ( 2018 - 02 - 09 ) 0.80 Jane and Rafael try to be friends , which works great until Jane goes on a date . Meanwhile , Petra and Jane 's new partnership is off to a rocky start , but they both know they have to find a way to work together somehow . Rogelio is convinced he has male post-partum depression , which puts him at odds with with celebrity Post-Partum Advocate River Fields ( guest star Brooke Shields ) on The Talk . Also , Alba finally confides in Jane why she turned down the marriage proposal , leaving Jane to help Alba in the romance department when Jane takes Alba to a sex store and buys her a vibrator . Elsewhere , Jane and Rafael agree to have sex , and Petra questions her own sexual identity when she starts to have sex dreams about Jane `` J.R. '' Ramos after the lawyer begins flirting with her . 75 11 `` Chapter Seventy - Five '' Gina Lamar Merigan Mulhern March 2 , 2018 ( 2018 - 03 - 02 ) 0.59 Jane is ready to move on from the Marbella Hotel issues and pursue writing full time , but an unexpected development forces her to consider returning to her teaching roots . Xiomara decides to go back to school where she listens in on an art history class , while Jane teaches a seminar about writing . Here , Jane finds out Jonathan Chavez , her former advisor , is taking advantage of another student of his . While trying to warn the student , Jane accidentally grabs her breasts to avoid falling over and , as a result , does n't get the teaching job she was applying for . Meanwhile , Petra gets arrested for Anezka 's murder due to the blackmailer planting the screws in her apartment , leading Jane Ramos to tell Petra that she was framing her because she was being threatened by an unknown blackmailer . Also , Rafael and Jane continue their affair . Elsewhere , Rogelio tries to get Eva Longoria to join his Passions of Santos reboot , but instead finds out the two are distant cousins . 76 12 `` Chapter Seventy - Six '' Melanie Mayron Leah Longoria March 9 , 2018 ( 2018 - 03 - 09 ) 0.58 Jane 's fixation on a negative review has left her with writer 's block , so Rogelio suggests she take a writing class to help . With the encouragement of Jane , Rafael is ready to dig into his past to find out who his real biological parents are . Meanwhile , Rogelio is trying to make things less about him and more about Xo , but he grows suspicious when he learns she is keeping a secret from him . But the secret Xo has been keeping is not what Rogelio suspected . Elsewhere , Petra tests the loyalty of her lawyer , Jane Ramos , by pretending to be romantically interested in her while trying to find out the identity of the blackmailer . After Petra confides in Jane R. of her feelings which are not fake , the two of them sleep together for the first time . 77 13 `` Chapter Seventy - Seven '' Gina Lamar Deirdre Shaw & Chantelle M. Wells March 16 , 2018 ( 2018 - 03 - 16 ) 0.59 When Jane , Rogelio and Alba learn what is going on with Xo who might have breast cancer , they all try to comfort her , but Xo insists they let her live her life . Meanwhile , Rogelio finally gets River Fields to agree to meet with him , but he learns that River has another plan in mind . Also , Rafael is on a mission to find his sister , who may have the information he has been looking for of his origins . Rafael visits Rose in prison to ask for her help , but he quickly turns her down when she makes a demand to know Luisa 's location . Also , Jane is upset with Petra after her twin girls spill a secret to Mateo about the Tooth Fairy being make believe . At the same time , Jane ' J.R. ' Ramos tells Petra after their romantic night together that she is not interested in a relationship due to the blackmailer paying a visit to Jane R. 's dementia - stricken mother in a rest home . At the end , the identity of Jane R 's blackmailer is finally revealed . 78 14 `` Chapter Seventy - Eight '' Justin Baldoni Valentina Garza & Micah Schraft March 23 , 2018 ( 2018 - 03 - 23 ) 0.65 Xo struggles to make an important decision on whether or not to get a full mastectomy and looks to Jane for guidance , leaving Rogelio feeling like an outsider . Alba is frustrated with Rogelio when it seems he is once again making his career a priority over Xo 's well being . Meanwhile , Petra turns to Rafael for advice on her love life after he finally learns about her new romance with J.R. 79 15 `` Chapter Seventy - Nine '' Gina Lamar Paul Sciarrotta April 6 , 2018 ( 2018 - 04 - 06 ) 0.53 80 16 `` Chapter Eighty '' Micah Schraft Micah Schraft April 13 , 2018 ( 2018 - 04 - 13 ) 0.57 Jane and Rafael start saving up to buy a one - bedroom apartment , but their plan falls apart when Mateo breaks an antique $4,000 clock . Rafael moves into the Villanueva house , but after Alba hits Mateo , the two get into a heated argument and Rafael moves back out . Xiomara starts chemotherapy , and makes a friend named Donna . The timeline moves forward nine weeks , and Donna dies of unforeseen consequences . Xiomara is shown not to be doing well due to the chemotherapy , Alba and Rafael are still in a fight and River and Rogelio are still arguing over the Passions of Santos adaptation . Alba breaks down to Jane , and she and Rafael make up . Xiomara bonds with River over telenovelas , after she accidentally eats a pot brownie , and she agrees to do the Passions of Steve and Brenda the Latin American way . Jane Ramos gets herself disbarred , and as a result gets the case against Petra thrown out . 81 17 `` Chapter Eighty - One '' Gina Lamar Jennie Snyder Urman & Paul Sciarrotta April 20 , 2018 ( 2018 - 04 - 20 ) 0.58 Rafael intends to propose to Jane , however Rose 's news unsettles him . Alba becomes a citizen and the Villanueva 's throw her a surprise party . Jane Ramos and Petra plan to move in together , until Jane finds out that Petra killed Anezka . As she walks out , Krishna tells her that her life is in danger . J.R. saves Petra 's life by shooting , and killing , the blackmailer . As Jane meets Rafael , Michael is revealed to be alive .
  Fact 1.0: The listing has columns for overall episode number, season episode number, title, director, writers, original air date, and U.S. viewers in millions.
  Fact 1.1: Chapter Sixty-Five is episode 65 overall, directed by Brad Silberling and written by Jennie Snyder Urman and Paul Sciarrotta.
  Fact 1.2: Chapter Sixty-Five first aired October 13, 2017, with 0.68 million U.S. viewers.
  Fact 1.3: Chapter Sixty-Five picks up immediately after where Season 3 left off.
  Fact 1.4: Jane is reunited with her first love, Adam, who found the love letter that Michael wrote.
  Fact 1.5: Jane is surprised but happy to see Adam.
  Fact 1.6: Xo and Alba are less thrilled that Adam is back.
  Fact 1.7: Rafael loses ownership of the hotel to his sister.
  Fact 1.8: As a result, Rafael finds himself living at Jane's house, broke and homeless.
  Fact 1.9: Jane and Rafael learn that the missing Petra has been found after she got away from Anezka.
  Fact 1.10: Jane and Rafael rush to Petra's side, but Anezka is nowhere to be found.
  Fact 1.11: When Petra returns to the hotel, Luisa thinks Petra is Anezka.
  Fact 1.12: Petra goes along with the game until the real Anezka returns.
  Fact 1.13: Xo and Rogelio's married life is interrupted when they confront the realization that Darci is having Rogelio's baby.
  Fact 1.14: Darci wants full custody of the baby.
  Fact 1.15: Chapter Sixty-Six is directed by Gina Lamar and written by Valentina Garza, Jessica O'Toole and Amy Rardin.
  Fact 1.16: Chapter Sixty-Six first aired October 20, 2017, with 0.61 million U.S. viewers.
  Fact 1.17: Jane's relationship with Adam helps bring out her 'fun side'.
  Fact 1.18: Adam's relationship with Jane reminds her that she is not just a mom.
  Fact 1.19: Adam gets a job opportunity out of state.
  Fact 1.20: Adam asks Jane's opinion on whether he should stay or go.
  Fact 1.21: Rogelio continues to face issues with his co-star at work.
  Fact 1.22: Rogelio's issues with his co-star are about to get even worse.
  Fact 1.23: Xo is helping to take care of Darci to prove that Rogelio is committed to being a father to the baby.
  Fact 1.24: Petra and Rafael begin working together to try to buy back the hotel from Luisa.
  Fact 1.25: Petra and Rafael run into trouble involving Anezka.
  Fact 1.26: Petra and Rafael's trouble involving Anezka causes tension between them.
  Fact 1.27: Chapter Sixty-Seven is directed by Fernando Sariñana and written by Carolina Rivera and Micah Schraft.
  Fact 1.28: Chapter Sixty-Seven first aired October 27, 2017, with 0.60 million U.S. viewers.
  Fact 1.29: Jane and Adam's relationship seems to be moving on.
  Fact 1.30: Jane thinks it might be time for Adam to meet her son Mateo.
  Fact 1.31: Rafael is not happy with the idea of Adam meeting Mateo.
  Fact 1.32: As the arrival of Darci's baby approaches, Rogelio's jealousy of Darci's new boyfriend increases.
  Fact 1.33: Xo encourages Rogelio to step up and be the father he was meant to be.
  Fact 1.34: Rafael is determined to get his hotel and life back from Luisa.
  Fact 1.35: Rafael will do whatever he has to do to get his hotel and life back, with or without Petra's help.
  Fact 1.36: Chapter Sixty-Eight is directed by Gina Lamar and written by Deirdre Shaw and Chantelle M. Wells.
  Fact 1.37: Chapter Sixty-Eight first aired November 3, 2017, with 0.69 million U.S. viewers.
  Fact 1.38: Jane becomes angry when she learns Rafael introduced Mateo to his new girlfriend, Katherine, without her consent.
  Fact 1.39: Jane thinks Katherine is a negative influence on both Rafael and Mateo.
  Fact 1.40: Jane turns to Petra for help with the situation.
  Fact 1.41: Rogelio and Darci struggle to choose a name for the baby.
  Fact 1.42: Xo is focused on opening her own dance studio.
  Fact 1.43: Xo decides to go into business with an unlikely foe.
  Fact 1.44: Chapter Sixty-Nine is episode 69 overall and season 5, directed by Stuart Gillard and written by Paul Sciarrotta.
  Fact 1.45: Chapter Sixty-Nine first aired November 10, 2017, with 0.65 million U.S. viewers.
  Fact 1.46: After Adam reveals his dating past to Jane, Jane becomes unsure and insecure.
  Fact 1.47: Jane's insecurity puts a strain on her relationship with Adam.
  Fact 1.48: Lina seeks Jane's advice about whether she should get married.
  Fact 1.49: Petra hears important information that could help Rafael get the hotel back.
  Fact 1.50: Xo asks Rogelio to consider having a vasectomy.
  Fact 1.51: Rogelio is very hesitant about the vasectomy.
  Fact 1.52: Chapter Seventy is episode 70 overall and season 6, directed by Melanie Mayron and written by Carolina Rivera and Micah Schraft.
  Fact 1.53: Chapter Seventy first aired November 17, 2017, with 0.61 million U.S. viewers.
  Fact 1.54: Jane's book is finally being released.
  Fact 1.55: Jane's plans to have a party at the local bookshop where she was inspired to be a writer hit a snag.
  Fact 1.56: Rogelio devises a plan to save the day for Jane.
  Fact 1.57: One of Rogelio's arch-rivals threatens to ruin the day.
  Fact 1.58: Rafael gets a new job, but it is not exactly what he was hoping for.
  Fact 1.59: Petra has come up with an interesting strategy to get the Marbella Hotel back from Luisa.
  Fact 1.60: Rafael wants nothing to do with Petra's scheme.
  Fact 1.61: Chapter Seventy-One is episode 71 overall and season 7, directed by Micah Schraft and written by Valentina Garza and Deidre Shaw.
  Fact 1.62: Chapter Seventy-One first aired December 8, 2017, with 0.65 million U.S. viewers.
  Fact 1.63: Anezka is revealed to have faked her death to see where her loyalties lie.
  Fact 1.64: Anezka initially trusts Petra after believing she saw Petra cry.
  Fact 1.65: Anezka threatens Petra after finding out Petra was not crying because of Anezka's fake death.
  Fact 1.66: Later, Anezka falls from Petra's apartment to her death, for real.
  Fact 1.67: Xiomara and Jane set off on Jane's book tour.
  Fact 1.68: Alba decides to tag along on Jane's book tour.
  Fact 1.69: After they get back, Jane and Rafael share a kiss.
  Fact 1.70: Chapter Seventy-Two is episode 72 overall and season 8, directed by Melanie Mayron and written by Carolina Rivera and Paul Sciarrotta.
  Fact 1.71: Chapter Seventy-Two first aired January 26, 2018, with 0.68 million U.S. viewers.
  Fact 1.72: Mateo starts kindergarten.
  Fact 1.73: Mateo invites a friend over to Xiomara and Rogelio's house.
  Fact 1.74: Jane and Rafael try to cover up that they are pretending to live in their house so they don't get kicked out.
  Fact 1.75: Jane and Rafael show mixed emotions following the kiss and try again, which Mateo notices.
  Fact 1.76: Mateo proclaims: 'Does this mean you're going to get married?'
  Fact 1.77: After Xo convinces Rogelio to attend couples therapy, Xo and Rogelio are both surprised by their emotions.
  Fact 1.78: Petra hires a lawyer named Jane Ramos, played by Rosario Dawson.
  Fact 1.79: Petra hires Jane Ramos to ensure Petra gets no charges pending Anezka's death.
  Fact 1.80: Rafael moves out of Jane's house into his new apartment.
  Fact 1.81: Chapter Seventy-Three is episode 73 overall and season 9, directed by Eric Lea and written by Valentina Garza and Chantelle M. Wells.
  Fact 1.82: Chapter Seventy-Three first aired February 2, 2018, with 0.67 million U.S. viewers.
  Fact 1.83: Jane and Rafael realize that the uncertain state of their relationship is stressing out Mateo.
  Fact 1.84: Jane and Rafael decide that they need to make a decision.
  Fact 1.85: Petra is on the outs with Jane and Rafael after the Luisa fiasco.
  Fact 1.86: Petra tries to figure out a way to get back into Jane and Rafael's good graces.
  Fact 1.87: Rogelio and Xo are both contemplating their career choices.
  Fact 1.88: Chapter Seventy-Four is episode 74 overall and season 10, directed by Gina Rodriguez and written by Micah Schraft and Paul Sciarrotta.
  Fact 1.89: Chapter Seventy-Four first aired February 9, 2018, with 0.80 million U.S. viewers.
  Fact 1.90: Jane and Rafael try to be friends, which works great until Jane goes on a date.
  Fact 1.91: Petra and Jane's new partnership is off to a rocky start.
  Fact 1.92: Petra and Jane both know they have to find a way to work together.
  Fact 1.93: Rogelio is convinced he has male post-partum depression.
  Fact 1.94: Rogelio's belief puts him at odds with celebrity Post-Partum Advocate River Fields on The Talk.
  Fact 1.95: River Fields is guest starred by Brooke Shields.
  Fact 1.96: Alba finally confides in Jane why she turned down the marriage proposal.
  Fact 1.97: Jane helps Alba in the romance department.
  Fact 1.98: Jane takes Alba to a sex store and buys her a vibrator.
  Fact 1.99: Jane and Rafael agree to have sex.
  Fact 1.100: Petra questions her own sexual identity.
  Fact 1.101: Petra starts to have sex dreams about Jane 'J.R.' Ramos after the lawyer begins flirting with her.
  Fact 1.102: Chapter Seventy-Five is episode 75 overall and season 11, directed by Gina Lamar and written by Merigan Mulhern.
  Fact 1.103: Chapter Seventy-Five first aired March 2, 2018, with 0.59 million U.S. viewers.
  Fact 1.104: Jane is ready to move on from the Marbella Hotel issues and pursue writing full time.
  Fact 1.105: An unexpected development forces Jane to consider returning to her teaching roots.
  Fact 1.106: Xiomara decides to go back to school, where she listens in on an art history class.
  Fact 1.107: Jane teaches a seminar about writing.
  Fact 1.108: Jane finds out Jonathan Chavez, her former advisor, is taking advantage of another student.
  Fact 1.109: While trying to warn the student, Jane accidentally grabs her breasts to avoid falling over.
  Fact 1.110: Because of this, Jane does not get the teaching job she was applying for.
  Fact 1.111: Petra gets arrested for Anezka's murder due to the blackmailer planting the screws in Petra's apartment.
  Fact 1.112: Jane Ramos tells Petra that she was framing Petra because Jane was being threatened by an unknown blackmailer.
  Fact 1.113: Rafael and Jane continue their affair.
  Fact 1.114: Rogelio tries to get Eva Longoria to join his Passions of Santos reboot.
  Fact 1.115: Rogelio finds out that he and Eva Longoria are distant cousins.
  Fact 1.116: Chapter Seventy-Six is episode 76 overall and season 12, directed by Melanie Mayron and written by Leah Longoria.
  Fact 1.117: Chapter Seventy-Six first aired March 9, 2018, with 0.58 million U.S. viewers.
  Fact 1.118: Jane's fixation on a negative review has left her with writer's block.
  Fact 1.119: Rogelio suggests Jane take a writing class to help.
  Fact 1.120: With Jane's encouragement, Rafael is ready to dig into his past to find out who his real biological parents are.
  Fact 1.121: Rogelio is trying to make things less about him and more about Xo.
  Fact 1.122: Rogelio grows suspicious when he learns Xo is keeping a secret from him.
  Fact 1.123: The secret Xo has been keeping is not what Rogelio suspected.
  Fact 1.124: Petra tests the loyalty of her lawyer, Jane Ramos, by pretending to be romantically interested in her.
  Fact 1.125: Petra is trying to find the identity of the blackmailer.
  Fact 1.126: Petra confides in Jane R. about her feelings, which are not fake.
  Fact 1.127: Petra and Jane R. sleep together for the first time.
  Fact 1.128: Chapter Seventy-Seven is episode 77 overall and season 13, directed by Gina Lamar and written by Deirdre Shaw and Chantelle M. Wells.
  Fact 1.129: Chapter Seventy-Seven first aired March 16, 2018, with 0.59 million U.S. viewers.
  Fact 1.130: Jane, Rogelio and Alba learn that Xo might have breast cancer.
  Fact 1.131: Jane, Rogelio and Alba try to comfort Xo.
  Fact 1.132: Xo insists they let her live her life.
  Fact 1.133: Rogelio finally gets River Fields to agree to meet with him.
  Fact 1.134: Rogelio learns that River has another plan in mind.
  Fact 1.135: Rafael is on a mission to find his sister, who may have information he has been looking for about his origins.
  Fact 1.136: Rafael visits Rose in prison to ask for her help.
  Fact 1.137: Rafael quickly turns Rose down when she demands to know Luisa's location.
  Fact 1.138: Jane is upset with Petra after Jane's twin girls spill a secret to Mateo about the Tooth Fairy being make-believe.
  Fact 1.139: Jane R. Ramos tells Petra she is not interested in a relationship after their romantic night together.
  Fact 1.140: The blackmailer paid a visit to Jane R.'s dementia-stricken mother in a rest home.
  Fact 1.141: The identity of Jane R.'s blackmailer is finally revealed at the end of the episode.
  Fact 1.142: Chapter Seventy-Eight is episode 78 overall and season 14, directed by Justin Baldoni and written by Valentina Garza and Micah Schraft.
  Fact 1.143: Chapter Seventy-Eight first aired March 23, 2018, with 0.65 million U.S. viewers.
  Fact 1.144: Xo struggles to make an important decision on whether or not to get a full mastectomy.
  Fact 1.145: Xo looks to Jane for guidance on the mastectomy decision.
  Fact 1.146: Xo's decision leaves Rogelio feeling like an outsider.
  Fact 1.147: Alba is frustrated with Rogelio when he once again seems to make his career a priority over Xo's well-being.
  Fact 1.148: Petra turns to Rafael for advice on her love life.
  Fact 1.149: Rafael finally learns about Petra's new romance with J.R.
  Fact 1.150: Chapter Seventy-Nine is episode 79 overall and season 15, directed by Gina Lamar and written by Paul Sciarrotta.
  Fact 1.151: Chapter Seventy-Nine first aired April 6, 2018, with 0.53 million U.S. viewers.
  Fact 1.152: Chapter Eighty is episode 80 overall and season 16, directed by Micah Schraft and written by Micah Schraft.
  Fact 1.153: Chapter Eighty first aired April 13, 2018, with 0.57 million U.S. viewers.
  Fact 1.154: Jane and Rafael start saving up to buy a one-bedroom apartment.
  Fact 1.155: Jane and Rafael's plan falls apart when Mateo breaks an antique $4,000 clock.
  Fact 1.156: Rafael moves into the Villanueva house.
  Fact 1.157: After Alba hits Mateo, Alba and Rafael get into a heated argument.
  Fact 1.158: Rafael moves back out after the argument.
  Fact 1.159: Xiomara starts chemotherapy.
  Fact 1.160: Xiomara makes a friend named Donna.
  Fact 1.161: The timeline moves forward nine weeks.
  Fact 1.162: Donna dies of unforeseen consequences.
  Fact 1.163: Xiomara is shown not to be doing well due to the chemotherapy.
  Fact 1.164: Alba and Rafael are still in a fight.
  Fact 1.165: River and Rogelio are still arguing over the Passions of Santos adaptation.
  Fact 1.166: Alba breaks down to Jane.
  Fact 1.167: Alba and Rafael make up.
  Fact 1.168: Xiomara bonds with River over telenovelas.
  Fact 1.169: Xiomara accidentally eats a pot brownie.
  Fact 1.170: Xiomara agrees to do the Passions of Steve and Brenda the Latin American way.
  Fact 1.171: Jane Ramos gets herself disbarred.
  Fact 1.172: As a result, Jane Ramos gets the case against Petra thrown out.
  Fact 1.173: Chapter Eighty-One is episode 81 overall and season 17, directed by Gina Lamar and written by Jennie Snyder Urman and Paul Sciarrotta.
  Fact 1.174: Chapter Eighty-One first aired April 20, 2018, with 0.58 million U.S. viewers.
  Fact 1.175: Rafael intends to propose to Jane.
  Fact 1.176: Rose's news unsettles Rafael.
  Fact 1.177: Alba becomes a citizen.
  Fact 1.178: The Villanueva family throws Alba a surprise party.
  Fact 1.179: Jane Ramos and Petra plan to move in together.
  Fact 1.180: Jane finds out Petra killed Anezka.
  Fact 1.181: As Jane walks out, Krishna tells her that her life is in danger.
  Fact 1.182: J.R. saves Petra's life by shooting and killing the blackmailer.
  Fact 1.183: As Jane meets Rafael, Michael is revealed to be alive.

Retrieved items:
<item 1>
Jane the Virgin (season 4)
Jane the Virgin ( season 4 ) - wikipedia Jane the Virgin ( season 4 ) Jump to : navigation , search Jane the Virgin ( season 4 ) Starring Gina Rodriguez Andrea
Navedo Yael Grobglas Justin Baldoni Ivonne Coll Elias Janssen Jaime Camil Anthony Mendez Country of origin United States No. of episodes 17 Release Original
network The CW Original release October 13 , 2017 ( 2017 - 10 - 13 ) -- April 20 , 2018 ( 2018 - 04 - 20 ) Season chronology ← Previous Season 3 Next → Season 5
List of Jane the Virgin episodes The fourth season of Jane the Virgin aired on The CW from October 13 , 2017 to April 20 , 2018 . The season consisted of 17
episodes and stars Gina Rodriguez as a young Latina university student accidentally artificially inseminated with sperm from her boss , Rafael Solano ( Justin
Baldoni ) .
</item 1>

<item 2>
Jane the Virgin (season 4)
Also , Rafael is on a mission to find his sister , who may have the information he has been looking for of his origins . Rafael visits Rose in prison to ask for
her help , but he quickly turns her down when she makes a demand to know Luisa 's location . Also , Jane is upset with Petra after her twin girls spill a secret
to Mateo about the Tooth Fairy being make believe . At the same time , Jane ' J.R. ' Ramos tells Petra after their romantic night together that she is not
interested in a relationship due to the blackmailer paying a visit to Jane R. 's dementia - stricken mother in a rest home . At the end , the identity of Jane R
's blackmailer is finally revealed . 78 14 `` Chapter Seventy - Eight '' Justin Baldoni Valentina Garza & Micah Schraft March 23 , 2018 ( 2018 - 03 - 23 ) 0.65
Xo struggles to make an important decision on whether or not to get a full mastectomy and looks to Jane for guidance , leaving Rogelio feeling like an outsider
.
</item 2>

<item 3>
Jane the Virgin (season 4)
Here , Jane finds out Jonathan Chavez , her former advisor , is taking advantage of another student of his . While trying to warn the student , Jane
accidentally grabs her breasts to avoid falling over and , as a result , does n't get the teaching job she was applying for . Meanwhile , Petra gets arrested
for Anezka 's murder due to the blackmailer planting the screws in her apartment , leading Jane Ramos to tell Petra that she was framing her because she was
being threatened by an unknown blackmailer . Also , Rafael and Jane continue their affair . Elsewhere , Rogelio tries to get Eva Longoria to join his Passions
of Santos reboot , but instead finds out the two are distant cousins . 76 12 `` Chapter Seventy - Six '' Melanie Mayron Leah Longoria March 9 , 2018 ( 2018 - 03
- 09 ) 0.58 Jane 's fixation on a negative review has left her with writer 's block , so Rogelio suggests she take a writing class to help . With the
encouragement of Jane , Rafael is ready to dig into his past to find out who his real biological parents are .
</item 3>

<item 4>
Jane the Virgin (season 4)
Rogelio is convinced he has male post-partum depression , which puts him at odds with with celebrity Post-Partum Advocate River Fields ( guest star Brooke
Shields ) on The Talk . Also , Alba finally confides in Jane why she turned down the marriage proposal , leaving Jane to help Alba in the romance department
when Jane takes Alba to a sex store and buys her a vibrator . Elsewhere , Jane and Rafael agree to have sex , and Petra questions her own sexual identity when
she starts to have sex dreams about Jane `` J.R. '' Ramos after the lawyer begins flirting with her . 75 11 `` Chapter Seventy - Five '' Gina Lamar Merigan
Mulhern March 2 , 2018 ( 2018 - 03 - 02 ) 0.59 Jane is ready to move on from the Marbella Hotel issues and pursue writing full time , but an unexpected
development forces her to consider returning to her teaching roots . Xiomara decides to go back to school where she listens in on an art history class , while
Jane teaches a seminar about writing .
</item 4>

<item 5>
Jane the Virgin (season 4)
However , both Xo and Alba are less thrilled that he is back . Meanwhile , Rafael loses ownership of the hotel to his sister and as a result , he finds himself
living at Jane 's house , broke and homeless . Jane and Rafael learn that the missing Petra has been found after she got away from Anezka and they rush to her
side , but Anezka is nowhere to be found . When Petra returns to the hotel , Luisa thinks that Petra is Anezka . Petra goes along with the game , until the real
Anezka returns . Elsewhere , Xo and Rogelio 's married life is interrupted when they confront the realization that Darci is having Rogelio 's baby and she wants
full custody . 66 `` Chapter Sixty - Six '' Gina Lamar Valentina Garza & Jessica O'Toole & Amy Rardin October 20 , 2017 ( 2017 - 10 - 20 ) 0.61 Jane 's
relationship with Adam helps bring out her `` fun side '' and reminds her that she is n't just a mom .
</item 5>

<item 6>
Jane the Virgin (season 4)
Also , Rafael moves out of Jane 's house into his new apartment . 73 9 `` Chapter Seventy - Three '' Eric Lea Valentina Garza & Chantelle M. Wells February 2 ,
2018 ( 2018 - 02 - 02 ) 0.67 When Jane and Rafael finally realize that the uncertain state of their relationship ( or lack of it ) is stressing out Mateo , they
decide that they need to make a decision . Meanwhile , Petra is on the outs with Jane and Rafael after the Luisa fiasco , so she tries to figure out a way to
get back into their good graces . Elsewhere , Rogelio and Xo are both contemplating their career choices . 74 10 `` Chapter Seventy - Four '' Gina Rodriguez
Micah Schraft & Paul Sciarrotta February 9 , 2018 ( 2018 - 02 - 09 ) 0.80 Jane and Rafael try to be friends , which works great until Jane goes on a date .
Meanwhile , Petra and Jane 's new partnership is off to a rocky start , but they both know they have to find a way to work together somehow .
</item 6>

<item 7>
Jane the Virgin (season 4)
Ramos Alfonso DiLuca as Jorge Garcia Special Guest Star ( edit ) Brooke Shields as River Fields Guest ( edit ) Francisco San Martin as Fabian Regalo del Cielo
Diane Guerrero as Lina Santillan Sharon Osbourne as herself Julie Chen as herself Sara Gilbert as herself Sheryl Underwood as herself Eve as herself Melanie
Mayron as Marlene Donaldson Eva Longoria as herself Adam Rodriguez as Jonathan Chavez Amy Brenneman as Donna Episodes ( edit ) See also : List of Jane the
Virgin episodes No . overall No. in season Title Directed by Written by Original air date U.S. viewers ( millions ) 65 `` Chapter Sixty - Five '' Brad
Silberling Jennie Snyder Urman & Paul Sciarrotta October 13 , 2017 ( 2017 - 10 - 13 ) 0.68 Picking up immediately after where Season 3 left off , Jane is
reunited with her first love , Adam , who found the love letter that Michael wrote . She is surprised , but happy , to see him . However , both Xo and Alba are
less thrilled that he is back .
</item 7>

<item 8>
Jane the Virgin (season 4)
69 5 `` Chapter Sixty - Nine '' Stuart Gillard Paul Sciarrotta November 10 , 2017 ( 2017 - 11 - 10 ) 0.65 After Adam reveals his dating past to Jane , she
becomes so unsure and insecure that it puts a strain on their relationship . Meanwhile , Lina seeks Jane 's advice about whether she should get married or not .
Petra hears some important information that could help Rafael get the hotel back . Also , Xo asks Rogelio to consider having a medical procedure ( vasectomy )
done , but he is very hesitant . 70 6 `` Chapter Seventy '' Melanie Mayron Carolina Rivera & Micah Schraft November 17 , 2017 ( 2017 - 11 - 17 ) 0.61 Jane 's
book is finally being released but her plans to have a party at the local bookshop where she was inspired to be a writer hit a snag . Always wanting to help
Jane , Rogelio devises a plan to save the day , but one of his arch - rivals threatens to ruin the day .
</item 8>

<item 9>
Jane the Virgin (season 4)
Meanwhile , Rogelio is trying to make things less about him and more about Xo , but he grows suspicious when he learns she is keeping a secret from him . But
the secret Xo has been keeping is not what Rogelio suspected . Elsewhere , Petra tests the loyalty of her lawyer , Jane Ramos , by pretending to be romantically
interested in her while trying to find out the identity of the blackmailer . After Petra confides in Jane R. of her feelings which are not fake , the two of
them sleep together for the first time . 77 13 `` Chapter Seventy - Seven '' Gina Lamar Deirdre Shaw & Chantelle M. Wells March 16 , 2018 ( 2018 - 03 - 16 )
0.59 When Jane , Rogelio and Alba learn what is going on with Xo who might have breast cancer , they all try to comfort her , but Xo insists they let her live
her life . Meanwhile , Rogelio finally gets River Fields to agree to meet with him , but he learns that River has another plan in mind .
</item 9>

<item 10>
Jane the Virgin (season 4)
Contents ( hide ) 1 Cast and characters 1.1 Main 1.2 Recurring 1.3 Special Guest Star 1.4 Guest 2 Episodes 3 Ratings 4 References Cast and characters ( edit )
Main ( edit ) Gina Rodriguez as Jane Gloriana Villanueva Andrea Navedo as Xiomara `` Xo '' Gloriana de la Vega Yael Grobglas as Petra Solano / Anezka Archuletta
Justin Baldoni as Rafael Solano Ivonne Coll as Alba Gloriana Garcia Elias Janssen as Mateo Gloriano Rogelio Solano Villanueva Jaime Camil as Rogelio de la Vega
Anthony Mendez as the Latin Lover Narrator ( voice over ) Recurring ( edit ) Tyler Posey as Adam Alvaro Yara Martinez as Dr. Luisa Alver Justina Machado as
Darci Factor Bridget Regan as Rose Solano Priscilla Barnes as Magda Shelly Bhalla as Krishna Rosario Dawson as Jane `` J.R. ''
</item 10>
=== END ITEM id=4df16687308afda8 ===

=== ITEM id=97a2435a72dc5bae ===
Gold sentences and facts:
Sentence 0: Republic of Doyle is a Canadian comedy - drama television series set in St. John 's , Newfoundland and Labrador which aired on CBC Television from January 6 , 2010 to December 10 , 2014 .
  Fact 0.0: Republic of Doyle is a Canadian comedy-drama television series.
  Fact 0.1: Republic of Doyle is set in St. John's, Newfoundland and Labrador.
  Fact 0.2: Republic of Doyle aired on CBC Television.
  Fact 0.3: Republic of Doyle aired from January 6, 2010 to December 10, 2014.

Retrieved items:
<item 1>
Republic of Doyle
Republic of Doyle - wikipedia Republic of Doyle Jump to : navigation , search Republic of Doyle Genre Comedy - drama Police procedural Created by Allan Hawco
Malcolm MacRury Perry Chafe Starring Allan Hawco Seán McGinley Lynda Boyd Krystin Pellerin Mark O'Brien Marthe Bernard Bob Cole Opening theme `` Oh Yeah '' by
Great Big Sea Original language ( s ) English No. of seasons 6 No. of episodes 77 ( list of episodes ) Production Executive producer ( s ) Allan Hawco , John
Vatcher , Michael Levine Location ( s ) St. John 's , Newfoundland and Labrador , Canada Cinematography Malcolm Cross Running time 44 minutes Production company
( s ) Take the Shot Productions Fireworks ( seasons 1 - 2 ) Content Film ( season 3 - 6 ) Distributor Trifecta Entertainment & Media ( United States ) Release
Original network CBC Television Picture format 1080i ( HDTV ) Original release January 6 , 2010 -- December 10 , 2014 External links Website Republic of Doyle
is a Canadian comedy - drama television series set in St.
</item 1>

<item 2>
Republic of Doyle
2015 TBA The Complete Series 77 September 6 , 2016 TBA International broadcasters ( edit ) Country Channel Premiere date South Africa Fox Entertainment 1 August
2010 United Kingdom Alibi 27 October 2010 Australia ABC2 ( season one ) 13th Street ( season two -- ) 6 November 2010 Brazil Globosat HD 11 November 2010
Estonia Fox Crime 11 September 2010 Serbia Fox Crime 14 October 2010 Bulgaria Fox Crime 4 December 2010 Poland TVP2 25 December 2012 Germany 13th Street 16
October 2013 United States CW Every Saturday and Sunday Netherlands 13th Street Until Friday 13 March 2015 , daily Monday - Friday Republic of Doyle is aired in
the USA on various stations via syndication . References ( edit ) Jump up ^ `` CBC winter season brings 3 new shows '' . cbc.ca , November 25 , 2009 . Jump up ^
`` Full Cast & Crew . '' IMDb . IMDb.com , n.d. Web . 17 Feb. 2014 . Jump up ^ `` Republic of Doyle renewed for 6th season '' .
</item 2>

<item 3>
Republic of Doyle
On September 6 , 2016 , eOne will release Republic of Doyle - The Complete Series on DVD in Canada . Acorn Media has released the first two seasons on DVD in
the USA . In Region 4 , Beyond Home Entertainment has released the first two seasons on DVD in Australia . DVD name Ep # Release dates Region 1 ( CAN ) Region 4
The Complete First Season 12 November 16 , 2010 May 4 , 2011 The Complete Second Season 13 July 19 , 2011 April 4 , 2012 The Complete Third Season 13 November
20 , 2012 TBA The Complete Fourth Season 13 October 1 , 2013 TBA The Complete Fifth Season 16 September 30 , 2014 TBA The Complete Sixth Season 10 April 21 ,
2015 TBA The Complete Series 77 September 6 ,
</item 3>

<item 4>
Republic of Doyle
Jump up ^ `` Republic of Doyle , Season 2 '' . Amazon.com . Retrieved 2017 - 10 - 04 . Jump up ^ `` REPUBLIC OF DOYLE - SEASON 1 Television Series Drama Comedy
'' . Beyondhomeentertainment.com.au . Retrieved 2012 - 11 - 09 . Jump up ^ `` REPUBLIC OF DOYLE - SEASON 2 Television Series Drama Comedy '' .
Beyondhomeentertainment.com.au . Retrieved 2012 - 11 - 09 . Jump up ^ `` Republic of Doyle '' . TVSA. 2010 - 08 - 01 . Retrieved 2010 - 08 - 01 . Jump up ^ ``
UKTV enters Republic of Doyle '' ( Press release ) . C21 Media . 2010 - 08 - 26 . Retrieved 2010 - 08 - 26 . Jump up ^ `` ABC2 Programming Airdate : Republic of
Doyle ( episode one ) '' . ABC Television Publicity. 2010 - 11 - 20 . Jump up ^ Newsome , Brad ( 6 January 2012 ) .
</item 4>

<item 5>
Republic of Doyle
Contents ( hide ) 1 Cast 1.1 Main 1.2 Recurring 2 Characters 3 Production 4 Season synopses 4.1 Season One 4.2 Season Two 4.3 Season Three 4.4 Season Four 4.5
Season Five 4.6 Season Six 5 Episodes 6 Reception 6.1 Ratings 7 Awards and nominations 7.1 Canadian Cinema Editors Awards 7.2 Canadian Comedy Awards 7.3
Canadian Screen Awards 7.4 Gemini Awards 8 DVD releases 9 International broadcasters 10 References 11 External links Cast ( edit ) Main ( edit ) Allan Hawco --
Jake Doyle ( Seasons 1 -- 6 ) Seán McGinley -- Malachy Doyle ( Seasons 1 -- 6 ) Lynda Boyd -- Rose Miller ( Seasons 1 -- 6 ) Krystin Pellerin -- Leslie Bennett
( Seasons 1 -- 6 ) Rachel Wilson -- Dr. Nikki Renholds ( Main , Season 1 ; Recurring , Seasons 2 -- 6 ) Mark O'Brien -- Desmond `` Des '' Courtney ( Seasons 1
-- 6 ) Marthe Bernard -- Katrina `` Tinny '' Doyle (
</item 5>

<item 6>
Republic of Doyle
Jump up ^ Newsome , Brad ( 6 January 2012 ) . `` Republic of Doyle , Friday , January 6 '' . The Sydney Morning Herald . Retrieved 17 January 2012 . Jump up ^
Daniel MacEachern ( 2013 - 04 - 17 ) . `` ' Republic of Doyle ' headed for U.S. '' The Telegram . Retrieved 2014 - 11 - 21 . External links ( edit ) Television
in Canada portal Official website Republic of Doyle on IMDb Retrieved from `` https://en.wikipedia.org/w/index.php?title=Republic_of_Doyle&oldid=837889899 ''
Categories : 2010 Canadian television series debuts CBC Television shows Culture of St.
</item 6>

<item 7>
Republic of Doyle
John 's , Newfoundland and Labrador Irish - Canadian culture Television shows set in Newfoundland and Labrador Television series produced in Newfoundland and
Labrador 2014 Canadian television series endings 2010s Canadian comedy - drama television series Television series by Entertainment One Fictional Canadian
people of Irish descent Hidden categories : Webarchive template archiveis links Talk Contents About Wikipedia Deutsch Français Polski Português Edit links This
page was last edited on 23 April 2018 , at 17 : 01 . About Wikipedia
</item 7>

<item 8>
Republic of Doyle
2012 ( 2012 - 01 - 11 ) April 4 , 2012 ( 2012 - 04 - 04 ) 13 January 6 , 2013 ( 2013 - 01 - 06 ) April 21 , 2013 ( 2013 - 04 - 21 ) 5 16 October 2 , 2013 ( 2013
- 10 - 02 ) February 5 , 2014 ( 2014 - 02 - 05 ) 6 10 October 15 , 2014 ( 2014 - 10 - 15 ) December 10 , 2014 ( 2014 - 12 - 10 ) Reception ( edit ) Ratings (
edit ) Season Timeslot ( ET ) # Ep . Premiered Ended TV season Viewers Date Premiere viewers Date Finale viewers Wednesday 9 : 00pm 12 January 6 , 2010 969,000
April 7 , 2010 ? ? 13 January 12 , 2011 1,038,000 April 6 , 2011 ? 2011 ? 13 January 11 , 2012 ?
</item 8>

<item 9>
Republic of Doyle
John 's , Newfoundland and Labrador which aired on CBC Television from January 6 , 2010 to December 10 , 2014 . The show stars Allan Hawco as private
investigator and former police officer Jake Doyle and Seán McGinley as his father , retired police officer Malachy Doyle . They partner as private investigators
with Rose Doyle , Malachy 's second wife , played by Lynda Boyd , in St. John 's . Krystin Pellerin stars as Constable / Sergeant Leslie Bennett , with Mark
O'Brien as Desmond `` Des '' Courtney , who works with the Doyles , and Marthe Bernard as Katrina `` Tinny '' Doyle , Jake 's niece and Malachy 's granddaughter
. Their cases involve them in all sorts of dealings -- not all of them on the right side of the law . The show was renewed on April 4 , 2014 , for a sixth and
final season .
</item 9>

<item 10>
Republic of Doyle
The character name is evocative of the character name , Bobby Crocker , and the actor who played him , Kevin Dobson , in the classic television series Kojak (
1973 -- 78 ) , starring Telly Savalas . Production ( edit ) The show is filmed in and around St. John 's , Newfoundland and Labrador . The show 's star , Allan
Hawco , is also executive producer / showrunner / co-creator and head writer of the show . Other producers include John Vatcher ( exec producer ) , Rob Blackie
, Perry Chafe ( co-creator and writer ) and Michael Levine . In 2013 , Republic of Doyle aired a fictional crossover with another CBC Television crime drama ,
Murdoch Mysteries . With the two shows set over 100 years apart , Hawco appeared on the November 25 , 2013 episode of Murdoch Mysteries as Jacob Doyle , a 19th
- century ancestor of Jake Doyle , while Yannick Bisson appeared on the January 29 , 2014 episode of Republic of Doyle as Detective Bill Murdoch , a 21st -
century descendant of his regular character Detective William Murdoch .
</item 10>
=== END ITEM id=97a2435a72dc5bae ===

=== ITEM id=54c934685446ff11 ===
Gold sentences and facts:
Sentence 0: In the United States , there is an Animal Welfare Act that was produced in 1966 . This law protects animals in acts of research , transportation , and sale . Generally , animals are protected from any torture , neglect , or killing . There have been many amendments made towards this act to keep it updated . In order to be charged for animal abuse , there has to be prosecution and a trial . While there is only one act covering the entire United States , there are more current laws surrounding animal rights , which vary by state . Even though there are policies and severe consequences , the laws are not normally enforced with as much force or timeliness as regular crime .
  Fact 0.0: In the United States, there is an Animal Welfare Act.
  Fact 0.1: The Animal Welfare Act was produced in 1966.
  Fact 0.2: The Animal Welfare Act protects animals in acts of research, transportation, and sale.
  Fact 0.3: Under the Animal Welfare Act, animals are generally protected from any torture, neglect, or killing.
  Fact 0.4: Many amendments have been made to the Animal Welfare Act to keep it updated.
  Fact 0.5: To be charged for animal abuse, there has to be prosecution and a trial.
  Fact 0.6: Only one act covers the entire United States.
  Fact 0.7: There are more current laws surrounding animal rights, and these laws vary by state.
  Fact 0.8: There are policies and severe consequences for animal abuse.
  Fact 0.9: Animal abuse laws are not normally enforced with as much force or timeliness as regular crime.

Retrieved items:
<item 1>
The U.S. animal-rights movement is dominated by professional nonprofits that favor moderate welfare reforms. Abolitionists, including Gary Francione, reject
policy reform as counterproductive and advocate nonviolent education, moral persuasion, and veganism to build antispeciesist culture and economically challenge
animal agriculture. A militant faction uses tactics such as property damage, animal releases, intimidation, and violence, aiming to change society through fear;
other activists criticize these tactics as counterproductive and likely to provoke repression. Reform-oriented groups
</item 1>

<item 2>
The animal rights movement began in Victorian England as aristocratic campaigns against mistreatment of workhorses and stray dogs. It includes mainstream
advocates and smaller faith-based and veganarchist factions, some of whom critique capitalism; internal diversity can strengthen movements. Once considered
fringe, it gained academic, legal, and professional support from the 1980s onward and also influenced punk subcultures. U.S. protections include the 1966 Animal
Welfare Act, though enforcement is often limited and laws vary by state
</item 2>

<item 3>
The modern animal-rights movement emerged in Britain in the early 1970s, building on influences including Upton Sinclair, Henry Salt, and vegan communities.
Oxford philosophy students Rosalind and Stanley Godlovitch, John Harris, and others argued for animals’ moral rights rather than mere kindness. Brigid Brophy’s
1965 critique of animal exploitation helped inspire them; Richard Ryder coined “speciesism” in 1970. Their influential 1972 book, *Animals, Men and
</item 3>

<item 4>
The animal-rights movement seeks to end speciesism, animals’ property status, and their use in food, research, clothing, and entertainment. Emerging in modern
form in 1970s Britain, it spans reformist nonprofits, abolitionists promoting veganism and education, and militant direct-action factions. Campaigns use
boycotts, undercover investigations, graphic imagery, online organizing, and legal advocacy. Although largely peaceful, some activists use threats or property
destruction, prompting criticism and repression. The movement has gained academic
</item 4>

<item 5>
Animal rights movement
In the United States , there is an Animal Welfare Act that was produced in 1966 . This law protects animals in acts of research , transportation , and sale .
Generally , animals are protected from any torture , neglect , or killing . There have been many amendments made towards this act to keep it updated . In order
to be charged for animal abuse , there has to be prosecution and a trial . While there is only one act covering the entire United States , there are more
current laws surrounding animal rights , which vary by state . Even though there are policies and severe consequences , the laws are not normally enforced with
as much force or timeliness as regular crime . Strategy and Tactical repertoires ( edit ) Use of New information communication technologies ( ICTs ) ( edit )
New media , such as the Internet and email , have been used by Animal Rights Movement actors and countermovement actors in a variety of capacities . Radical
factions in the movement rely on websites , blogs , podcasts , videos , and online forums to engage in vegan outreach and other mobilization efforts and build
alliances , thus overcoming exclusion by dominant factions . The use of Internet has allowed the animal rights movement to spread transnationally .
</item 5>

<item 6>
Animal rights movement
Other early influences include : Upton Sinclair 's 1906 novel The Jungle , which drew attention to obscured slaughterhouse operations ; Henry Salt 's treatises
on nonhuman animal rights ; which drew from human abolitionist arguments for recognizing personhood of people considered to be property ; and the Alcott House
of New England , a community serving as a stop on the Underground Railroad and requiring its residents to eat a vegan diet . Philosopher Peter Singer The
contemporary movement is regarded as having been founded in the UK in the early 1970s by a group of Oxford university post-graduate philosophy students , now
known as the `` Oxford Group '' . The group was led by Rosalind and Stanley Godlovitch , graduate students of philosophy who had recently become vegetarians .
The Godlovitches met John Harris and David Wood , also philosophy graduates , who were soon persuaded of the arguments in favour of animal rights and themselves
became vegetarian . The group began to actively raise the issue with pre-eminent Oxford moral philosophers , including Professor Richard Hare , both personally
and in lectures . Their approach was based not on sentimentality ( `` kindness to dumb animals ' ) , but on the moral rights of animals .
</item 6>

<item 7>
Animal rights movement
, the Animal Enterprise Terrorism Act ) and does not economically or politically challenge extant systems . Smaller factions include groups focused around faith
- based animal rights theory and veganarchists , whose approach is characterized by a critique of capitalism on the grounds that it has led to mass nonhuman ,
human , and environmental exploitation . Such factionalizing , researchers have pointed out , is common to social movements and plays a role in sustaining their
health . History ( edit ) Further information : 20th century : Increase in animal use ; animal rights movement , Gary Francione , Tom Regan , Richard D. Ryder ,
and Peter Singer The Animal Rights Movement traces back to the animal protection movement in Victorian England , which was initiated by aristocratic moral
crusaders in response to the poor treatment of urban workhorses and stray dogs .
</item 7>

<item 8>
Animal rights movement
It has been noted that the power of the animal rights movement in the United States is centralized in professionalized nonprofit organizations that prefer
moderated goals of welfare reform and tend to be isomorphic . The abolitionist faction diagnoses the problem as one of nonhuman animal use , offers a prognosis
including vegan outreach , and employ action frames including a rational and theoretic focus . Gary Francione , a leader in abolitionism , formed his approach
in response to the traditional movement 's focus on policy reform . Members of the abolitionist faction view policy reform as counterproductive and rely on
nonviolent education and moral suasion in their repertoire of contention . They see the promotion of veganism as a low - opportunity cost means of creating an
antispeciesist culture and posing an economic threat to the animal agriculture industrial complex . The direct action or militant faction includes in its
repertoire of contention property damage , open releases , intimidation , and direct violence . With these , they offer a prognosis to the problem of animal use
that focuses on changing society through force and fear . Animal rights actors often reject this faction , pointing to violence as a counterproductive tactic
that invites repression ( e.g.
</item 8>

<item 9>
Animal rights movement
This association continues on into the 21st century , as evidenced by the prominence of vegan punk events such as Fluff Fest in Europe . Michael Socarras of
Greenberg Traurig told the Association of American Medical Colleges : `` There is a very important shift under way in the manner in which many people in law
schools and in the legal profession think about animals . This shift has not yet reached popular opinion . However , in ( the U.S. ) , social change has and can
occur through the courts , which in many instances do not operate as democratic institutions . Therefore , the evolution in elite legal opinion is extremely
significant ... '' Philosophical and legal aims ( edit ) Further information : Main animal rights philosophical approaches and Veganarchism The movement aims to
include animals in the moral community by putting the basic interests of non-human animals on an equal footing with the basic interests of human beings . A
basic interest would be , for example , not being made to suffer pain on behalf of other individual human or non-human animals . The aim is to remove animals
from the sphere of property and to award them personhood ; that is , to see them awarded legal rights to protect their basic interests .
</item 9>

<item 10>
Animal rights movement
An inclusive movement with strong group solidarity would decrease opportunity costs associated with participating ( e.g. , social stigmatization , lack of
alternatives , legal persecution ) and thus serve to increase and sustain participation in the movement . Current status of the movement ( edit ) The movement
is no longer viewed as hovering on the fringe . In the 1980s and 1990s , it was joined by a wide variety of academics and professionals , including lawyers ,
physicians , psychologists , veterinarians , and former vivisectionists , and is now a common subject of study in philosophy departments in Europe and North
America . Animal law courses are taught in 92 out of 180 law schools in the U.S. , and the movement has gained the support of senior legal scholars , including
Alan Dershowitz and Laurence Tribe of Harvard Law School . Chapters of animal rights law have been created in several state bar associations , and resolutions
related to animal rights are regularly proposed within the American Bar Association . In the 1980s , the movement became associated with punk subculture and
ideologies , particularly straight edge hardcore punk in the United States and anarcho - punk in the United Kingdom .
</item 10>
=== END ITEM id=54c934685446ff11 ===

=== ITEM id=91e715648de0711c ===
Gold sentences and facts:
Sentence 0: Songwriter ( s ) Kirsty MacColl / Philip Rambow
  Fact 0.0: The songwriters of the song are Kirsty MacColl and Philip Rambow.
Sentence 1: `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy ... ( country version ) '' Released May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 )
  Fact 1.0: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is a single by Kirsty MacColl.
  Fact 1.1: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is from the album Desperate Character.
  Fact 1.2: The single's A-side is 'There's a Guy Works Down the Chip Shop Swears He's Elvis'.
  Fact 1.3: The single's B-side is 'Hard To Believe There's a Guy ... (country version)'.
  Fact 1.4: The single was released in May 1981.
  Fact 1.5: The single's format is 7-inch vinyl.
  Fact 1.6: The single's genre is country pop.
  Fact 1.7: The single's label is Polydor.
  Fact 1.8: The single's songwriters are Kirsty MacColl and Philip Rambow.
  Fact 1.9: The single's producer is Barry Farmer.
  Fact 1.10: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981).
  Fact 1.11: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' (1981).
  Fact 1.12: The Kirsty MacColl singles chronology includes 'See That Girl' (1981).
  Fact 1.13: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981) a second time.
  Fact 1.14: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop Swears He's Elvis' (1981).
  Fact 1.15: The Kirsty MacColl singles chronology includes 'See That Girl' (1981) a second time.

Retrieved items:
<item 1>
There's a Guy Works Down the Chip Shop Swears He's Elvis
`` Walking Down Madison '' `` Soho Square '' With The Pogues `` Fairytale of New York '' Related articles Ewan MacColl Steve Lillywhite Retrieved from ``
https://en.wikipedia.org/w/index.php?title=There%27s_a_Guy_Works_Down_the_Chip_Shop_Swears_He%27s_Elvis&oldid=799608311 '' Categories : 1981 singles 1981 songs
Elisabeth Andreassen songs Kirsty MacColl songs Songs written by Kirsty MacColl Songs about Elvis Presley Polydor Records singles Hidden categories : Articles
lacking reliable references from March 2008 All articles lacking reliable references Talk About Wikipedia Svenska Edit links This page was last edited on 8
September 2017 , at 19 : 12 . About Wikipedia
</item 1>

<item 2>
Where Have All the Flowers Gone?
President The Weavers The Weavers at Carnegie Hall The Weavers at Carnegie Hall Vol. 2 The Weavers : Was n't That a Time ! Other Precious Friend ( with Arlo
Guthrie ) Related Discography People 's Songs Rainbow Quest Sing Out ! Pete Seeger : The Power of Song Hudson River Sloop Clearwater The Great Hudson River
Revival We Shall Overcome : The Seeger Sessions Family Toshi Seeger ( wife ) Mika Seeger ( daughter ) Tao Rodríguez - Seeger ( grandson ) Charles Seeger (
father ) Ruth Crawford Seeger ( stepmother ) Mike Seeger ( half - brother ) Peggy Seeger ( half - sister ) Alan Seeger ( uncle ) Ewan MacColl ( brother - in -
law ) John Cohen ( brother - in - law ) Kirsty MacColl ( stepniece ) Marlene Dietrich Studio albums Marlene Dietrich Overseas ( 1951 ) Die neue Marlene ( 1965 )
Marlene singt Berlin ,
</item 2>

<item 3>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's
Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' was Kirsty MacColl 's first hit single ,
charting in the UK at # 14 in 1981 , and remaining in the charts for nine weeks . It reached # 9 in the Irish charts . A promotional version was released in the
USA , which instead referenced a truck stop as opposed to the very British chip shop , but this was only available to DJs and did not get a full release .
Norwegian singer Elisabeth Andreasson covered the song on her 1981 country album Angel of the Morning , with lyrics in Swedish by Hasse Olsson as `` Killen ner
' på Konsum svär att han är Elvis '' ( `` The guy down Konsum '' swears he 's Elvis ) .
</item 3>

<item 4>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Retrieved 2014 - 10 - 30 . Jump up ^ Information at Svensk mediedatabas External links ( edit ) freeworld - Kirsty MacColl fansite Kirsty MacColl Studio albums
Desperate Character Kirsty MacColl Kite Electric Landlady Titanic Days Tropical Brainstorm Compilations Galore Notable singles `` They Do n't Know '' `` Keep
Your Hands Off My Baby '' `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' `` You Still Believe in Me '' `` Terry '' `` A New England '' `` Days
'' `` Do n't Come The Cowboy With Me Sonny Jim ! ''
</item 4>

<item 5>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Track listing ( edit ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( K. MacColl / P. Rambow ) `` Hard to Believe '' ( K. MacColl ) `` There
's a Guy Works Down the Chip Shop Swears He 's Elvis ( country version ) '' ( K. MacColl / P. Rambow ) References ( edit ) Jump up ^ `` There 's a Guy Works
Down the Chip Shop Swears He 's Elvis - Kirsty MacColl '' . Retrieved 2014 - 10 - 30 . Jump up ^ `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis 7
'' single - Kirsty MacColl `` . Retrieved 2014 - 10 - 30 .
</item 5>

<item 6>
A Whiter Shade of Pale
You 're a Rich Man '' by The Beatles Procol Harum Gary Brooker Geoff Dunn Matt Pegg Josh Phillips Geoff Whitehorn Dave Ball Graham Broad Dave Bronze Pete Brown
Mark Brzezicki Alan Cartwright Chris Copping Matthew Fisher Mick Grabham Bobby Harrison David Knights Dee Murray Keith Reid Tim Renwick Ray Royer Don Snow Peter
Solley Henry Spinetti Jerry Stevenson Robin Trower Ian Wallace B.J.
</item 6>

<item 7>
All I Want for Christmas Is My Two Front Teeth
The song was introduced in 1948 by singing group The Satisfiers on Perry Como 's radio show , and originally recorded by Spike Jones & His City Slickers on
December 6 , 1947 , with lead vocal by George Rock . That version reached the top of the pop charts in 1948 , and again in 1949 . The song has also been
recorded by numerous other singers and performers , including Ray Stevens , George Strait , Danny Kaye with The Andrews Sisters , Urbie Green , The Platters ,
Dread Zeppelin , The Kelly Family , Nat King Cole ( reportedly Gardner 's favorite version ) , The Chipmunks , the Hampton String Quartet , The Three Stooges
and Count von Count of Sesame Street . In the versions by Spike , The Chipmunks ( Theodore on lead vocals ) , and Ray Stevens , the song ends with the performer
shouting the words `` Happy New Year ! ! ! '' Homer and Jethro parodied the song as `` All I Want for Christmas Is My Upper Plate '' , on their 1968 album Cool
Crazy Christmas .
</item 7>

<item 8>
There's a Guy Works Down the Chip Shop Swears He's Elvis
There 's a Guy Works Down the Chip Shop Swears He 's Elvis - wikipedia There 's a Guy Works Down the Chip Shop Swears He 's Elvis Jump to : navigation , search
This article relies too much on references to primary sources . Please improve this by adding secondary or tertiary sources . ( March 2008 ) ( Learn how and
when to remove this template message ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate
Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy . ( country version ) '' Released
May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles
chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop ,
</item 8>

<item 9>
Bad Boys (Inner Circle song)
203 . ISBN 1 - 894663 - 96 - 9 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Inner Circle Ian Lewis Roger Lewis Bernard `` Touter ''
Harvey Michael Sterling Lancelot Hall Trevor `` Skatta '' Bonnick Michael `` Ibo '' Cooper Stephen `` Cat '' Coore William Stewart William `` Bunny Rugs ''
Clarke Irvin `` Carrot '' Jarrett Milton `` Prilly '' Hamilton Charles Farquharson Calvin McKenzie Funky Brown Jacob Miller Lester Adderley Norman Grant Rick
Hunt Calton Coffie Dave Gonzales Kris Bentley Jr . Jazz Albums Bad to the Bone ( 1992 ) Singles `` Bad Boys '' `` Sweat ( A La La La La Long ) '' `` Games
People Play '' Related Discography Retrieved from `` https://en.wikipedia.org/w/index.
</item 9>

<item 10>
Lord Have Mercy on the Working Man
Critical reception ( edit ) Teresa M. Walker , in her review for the Gainesville Sun , said that with the assistance from superstars on the final chorus , the
song `` should shoot up the charts . '' Dave Larsen of the Dayton Daily News cited it as one of the stronger tracks on the album , saying that the album ``
works best when Tritt sticks with the populist approach . '' Alanna Nash of Entertainment Weekly said of the song , `` Tritt finds a solid image for his laborer
's lament ( ... ) But the melody is so slight that he resorts to an acoustic arrangement that evokes Jimmie Rodgers and the young Roy Acuff . '' Deborah Evans
Price , of Billboard magazine gave the song a mixed review , saying that while the final chorus of song features famous country artists , it was too bad that
the entire song is n't as strong as the last verse . Music video ( edit ) Jack Cole directed the song 's music video .
</item 10>
=== END ITEM id=91e715648de0711c ===

=== ITEM id=1324176cb8596335 ===
Gold sentences and facts:
Sentence 0: No . overall No. in series Title Vehicles tested Special guests Original air date 14 `` Past , Present , or Future '' Lamborghini Aventador S , Honda NSX , Rimac Concept One David Hasselhoff , Ricky Wilson 8 December 2017 ( 2017 - 12 - 08 ) The tent is now in the Cotswolds and will remain there for the duration of the series . To determine whether the past , present , or future of motoring is best , the presenters visit Switzerland with Clarkson in a petrol powered Lamborghini Aventador S representing the past , May in a hybrid powered Honda NSX representing the present , and Hammond in an all - electric Rimac Concept One representing the future . After starting at the top of the Gotthard Pass , Hammond brings the trio to a stay at the Park Weggis ( de ) health retreat and visit the Swiss Museum of Transport and other museums in Lucerne close to a charging station for the Rimac car . May and Clarkson reject this and go to a hill climbing race at Hemberg instead . As Clarkson can not fit into the Aventador whilst wearing a mandatory racing helmet , Clarkson sends a Lamborghini test driver in his place , thus setting the fastest time of all three . Hammond crashes after completing the course , destroying the Rimac in a fire . Footage taken from YouTube shows the wreckage being extinguished and Hammond being airlifted by the Swiss Air - Rescue ( REGA ) to hospital . The presenters discuss some inapropriate new features within cars on Conversation Street . Celebrity Brain Crash is replaced by Celebrity Face Off , in which Ricky Wilson beats David Hasselhoff to determine who the `` fastest former talent show host '' is . 15 `` The Falls Guys '' Ford GT , Mercedes AMG GT R Kevin Pietersen , Brian Wilson 15 December 2017 ( 2017 - 12 - 15 ) The presenters race from Central Park in Manhattan , New York City , to an observation tower overlooking the Niagara Falls . Clarkson drives the new Ford GT whilst May and Hammond ( on crutches ) use public transport , taking the New York City Subway , AirTrain JFK , a JetBlue Embraer E190 flight , plus two Niagara Frontier Transportation Authority buses . Despite Hammond and May 's flight arriving 15 minutes early , Clarkson narrowly wins the race . For Conversation Street the presenters discuss sponsorship ideas . Clarkson reviews the Mercedes AMG GT R , after which new test driver , Abbie Eaton , takes it for a timed lap around the Eboladrome . In `` Celebrity Face Off '' , Kevin Pietersen of South Africa beats Brian Wilson of the U.S. as the `` fastest person who makes a living throwing and hitting and catching small balls '' . 16 `` Bah Humbug - atti '' Kia Stinger GT , Bugatti Chiron Hugh Bonneville , Casey Anderson 22 December 2017 ( 2017 - 12 - 22 ) The presenters begin with Christmas tree lights made from vehicle parts and , Starting with Scalextric , they exchange humorous and inappropriate Christmas gifts in Conversation Street . May travels to Majorca to race a Kia Stinger GT against a pair of longboard riders along a mountain road , with May winning the race despite him racing uphill and the riders downhill . Hammond and May attempt to relieve lunchtime boredom by inventing `` lunchtime office car park racing '' at Houghton Hall business park in the Luton / Dunstable Urban Area . After brief interviews in which both share their own ownership of a Tesla Model S , Hugh Bonneville , representing the UK , and Casey Anderson , representing the United States , compete during `` Celebrity Face Off '' as the `` fastest person who lives with a bear '' , with Anderson winning by setting the faster lap time . Clarkson experiments with the jet set lifestyle , starting in the morning from the Da Vinci super-yacht moored in Saint - Tropez and then driving a Bugatti Chiron to L'Alpe De Venosc France for skiing at lunchtime , stopping briefly for a drag race against a BMW M4 at Gap -- Tallard Airport to give viewers a perspective on how fast the Chiron is . After leaving the Alps , Clarkson then pays an homage to the 1976 film C'était un rendez - vous , by speeding across Turin and reaching the Piazza Carignano ( it ) in 9 minutes . 17 `` Unscripted '' McLaren 720S , Audi TT RS , Ariel Nomad Michael Ball , Alfie Boe , Silvija Jurin 29 December 2017 ( 2017 - 12 - 29 ) The presenters begin by having a year - ending awards ceremony in Conversation Street . Hammond heads to the Eboladrome to test the McLaren 720S after which Eaton sets a lap time . Since The Grand Tour has been criticised for being too `` scripted '' , the presenters head to Zadar county and the island of Pag in Croatia to make a completely `` unscripted '' film . This results in Clarkson turning up in the Audi TT RS , Hammond turning up in the not at all comparable Ariel Nomad , and May turning up in an old Lada Riva which he turns into a fire engine . They go their separate ways though Clarkson and Hammond end up comparing their cars in a drag race , a rally stage ( both of which are won by Clarkson ) , visiting Maslenica , the Plitvice Lakes National Park , and following the D106 road until running out of road at Žigljen ferry port . Due to a lack of fires , May has to setup his own ones until Clarkson and Hammond setup one that proves too difficult for May , which the real fire brigade manage to put out with a firefighting aircraft , destroying May 's Lada fire engine in the process . Michael Ball and Alfie Boe compete in `` Celebrity Face Off '' with Ball setting the faster time and becoming the `` fastest classical singer with some kind of interesting connection to the British motor industry '' . 18 5 `` Up , Down and Round the Farm '' Ripsaw EV2 , Volkswagen Up ! GTI Bill Bailey , Dominic Cooper , Mark Higgins 5 January 2018 ( 2018 - 01 - 05 ) The presenters introduce some of the new cars for 2018 in Conversation Street . Hammond heads to Dubai to test a Sandrail dune buggy in the desert , followed by the Ripsaw `` high - end luxury tank '' , visiting the Dubai Mall . May reviews the Volkswagen Up ! GTI at the Eboladrome and in thick fog Eaton records a record slow time . Guests Bill Bailey and Dominic Cooper discuss their shared ownership of old French cars before racing each other in greasy / wet conditions with Cooper becoming the `` fastest person who has a failed former career in a band '' . Clarkson creates a Ken Block - style `` Farmkhana '' film using an old Subaru Impreza WRX STi around Hare Farm , near Rye , with bullet time photography . Hammond and May show the audience that Clarkson 's driving was mostly fake by sharing a `` making - of '' consisting of outtakes and behind - the - scenes footage featuring rally driver Mark Higgins , producer Phil Churchward , plus camera operator Ben Joiner breaking the fourth wall . Clarkson suggests that the whole thing is like sawing a woman in half during a magic show , and that really the audience do n't want to know . 19 6 `` Jaaaaaaaags '' None Luke Evans , Kiefer Sutherland 12 January 2018 ( 2018 - 01 - 12 ) The presenters head to the area around Grand Junction , Colorado , staying at the Gateway Canyons Resort & Spa , and eventually travelling to Telluride , Colorado with a variety of old Jaguar cars to establish the manufacturer 's reputation for reliability with Clarkson buying an XJR , May taking an XK8 convertible and Hammond bringing a Mark X . During the course of the trip the cars are driven around a dirt track ( where Hammond 's car is badly damaged , and is replaced by an XJ6 ) , along the runway at Telluride Regional Airport , ( where May 's car is destroyed , and is replaced by an XJS ) and down the slopes of the Telluride Ski Resort . At Hanging Flume overlook , Clarkson experiences a traffic stop by a Mesa County police officer and reads the `` Her Britannic Majesty '' text from his British passport . Despite Clarkson 's car being the best , May admitting his car had poor brakes deemed the challenge pointless . The presenters discuss the annoyance of Firemen cutting the roofs of cars in Conversation Street . Kiefer Sutherland beats Luke Evans to become `` the fastest actor with a middle name of George '' . 20 7 `` It 's a Gas , Gas , Gas '' Lamborghini Huracán Performante Anthony Joshua , Bill Goldberg 19 January 2018 ( 2018 - 01 - 19 ) The presenters discuss some new supercars and how handbrake turns impress women within the Conversation Street Segment . Hammond tests the Lamborghini Huracán Performante at the Eboladrome beating May 's Ferrari 458 Speciale in a drag race before Eaton takes it round the track setting a fast time . May and Hammond then look at ingenious ways of filling up with fuel instead of stopping at a fuel station . Anthony Joshua becomes `` the fastest man who makes a living out of punching or strangling someone '' beating Bill Goldberg , with both previously owning cars from General Motors . Clarkson then talks about the Rivalry in rallying between Audi with the Quattro and Lancia with the 037 . They conclude with a `` terrible disappointment '' that Lancia is no longer around . 21 8 `` Blasts from the Past '' Jaguar XK SS , Aston Martin DB4 GT , Honda Civic Type R , Ford GT Stewart Copeland , Nick Mason 26 January 2018 ( 2018 - 01 - 26 ) The presenters head to the City of Pau , Pyrénées - Atlantiques , France , and home of the Pau Grand Prix on the Circuit de Pau - Ville ( fr ) street course . At Palais Beaumont ( fr ) Hammond arrives with a Jagaur XK SS and Clarkson in an Aston Martin DB4 GT ; both being modern builds of older designs from the 1950s `` back - catalogues '' of Jaguar and Aston Martin . They both race around the Streets of Pau before May turns up in a modern Honda Civic Type R to prove that new cars are better . All three depart to drive south over the Pyrenees mountain range , with Clarkson and Hammond experiencing difficulties in dense fog . Clarkson recants a previous incident from 2013 -- when Clarkson and Hammond 's driving licences were confiscated whilst filming The Perfect Road Trip in France -- and tries to goad May into speeding on the Autopista AP - 2 . At Barcelona the presenters race around the abandoned Autódromo de Sitges - Terramar and the extreme banking on its corners , with May winning , and Hammond abstaining following his history of crashes . During Conversation Street the presenters discuss names for gritters . At the Eboladrome , Clarkson retests the Ford GT following the Niagara Falls race , with Eaton setting a test lap . Nick Mason goes on to record a faster lap than Stewart Copeland to become `` the fastest drummer for a band beginning with the letter p '' . 22 9 `` Breaking , Badly '' Jaguar XJ220 , Bugatti EB 110 Super Sport Dynamo , Penn & Teller 2 February 2018 ( 2018 - 02 - 02 ) The presenters discuss new sport cars by various companies , jumpers and the speed of pigeons vs blackbirds during Conversation Street . Clarkson unsuccessfully sets out to prove that mankind has not significantly improved over the last 25 years by testing the Jaguar XJ220 and the Bugatti EB 110 Super Sport , the fastest supercars from the early 90s , both of which set slow times at the Eboladrome when driven by Eaton . The trio tries to break the UK speed record for amphibious vehicles by building one of their own . Clarkson oversees his idea of combining a Rolls - Royce Spey turbofan engine from a Nimrod jet and a speedboat with a Suzuki Jimny . However on a nearby lake , Clarkson and May have to abort their run after achieving only 3 mph ( 5 km / h ) and nearly sinking . Hammond and May then go on to build a vehicle of their own by combining a Bond Bug with the engine of a personal water craft nicknaming it the `` Pond Bug '' . Competing on Coniston Water during Coniston Power Boat Records Week , Clarkson , being the only one with a speedboat licence , operates the `` Pond Bug '' and achieves an average of 47.81 mph ( 76.94 km / h ) , setting a new UK record . After performing a series of magic tricks , Dynamo and Penn & Teller compete to become `` the fastest magician in the world '' , with Penn & Teller prevailing on a snowy track . 23 10 `` Oh , Canada '' Alfa Romeo Stelvio Quadrifoglio , Range Rover Velar P380 , Porsche Macan Turbo Performance Pack , Tesla Model X , Ford F150 Raptor Rory McIlroy , Paris Hilton 9 February 2018 ( 2018 - 02 - 09 ) The presenters head to Canada to prove that small SUV 's are useless with Clarkson picking an Alfa Romeo Stelvio Quadrifoglio , May choosing a Range Rover Velar P380 and Hammond taking a Porsche Macan Turbo Performance Pack . They are shown to be slower round the Area 27 circuit than a BMW M3 , not practical enough to carry Canadian Newfoundland dogs , and slower round a barrel course than a horse : concluding that SUVs are pointless . Despite this , executive producer Andy Wilman informs via text that they will race their SUVs whilst each towing a boat up a mountain . May abandons his Velar for his boat , which sinks , causing him to lose the race . Because Clarkson 's Stelvio has no tow bar he cheats and exchanges it for a Ford F150 Raptor , ultimately arriving first with the presenters concluding that the Raptor was better than all the SUVs being tested . On Conversation Street the presenters talk about new SUVs and vans . Clarkson reviews the Tesla Model X at the Eboladrome and on the road , accompanied by some lawyers , ostensibly to avoid Tesla pursuing any legal action , following the previous Tesla v. Top Gear court cases . Rory McIlroy and Paris Hilton discuss their car collections before McIlroy beats Hilton to become the `` World 's Fastest Golfing Enthusiast '' . 24 11 `` Feed the World '' None None 16 February 2018 ( 2018 - 02 - 16 ) The presenters head to Mozambique on a mission to feed the people of Bingo , an isolated town in the west of the country , with fish from the country 's capital of Maputo , covering a distance of approximately 200 miles . To do this , each of them attempt to find the best transport solution : May buys a Mercedes - Benz Estate and modifies it by fitting a Perspex tank filled with sea water in which to keep the fish alive ; Clarkson buys a Nissan Hardbody truck which he fits with a generator and an ice machine in an attempt to keep the fish frozen and Hammond buys a TVS Star motorcycle and devises a rack which he fits to the bike 's rear , inspired by a local method of drying and salting the fish to preserve them . After unsuccessfully trying to catch fish using local methods , they buy fish at a local market . Early in the journey , Clarkson 's truck crashes into May 's car upon rescuing it , damaging its radiator grille . Deeper into rural Mozambique , May constantly soaks himself with sea water from his ill - fitted tank and , due to his damaged radiator grille , suffers multiple breakdowns due to water intrusion . Hammond loses a large number of his fish on the bumpy roads , constantly gets stuck in thick mud and endlessly falls from his motorbike . Clarkson suffers very few issues except when his ice machine breaks , causing him to make small adjustments to his truck in order to smoke the fish with his exhaust . Upon May 's car breaking down in the middle of a lake , despite Hammond and Clarkson crossing successfully , Clarkson cuts off the front end of the Mercedes and tows what remains of the rear portion using his truck - much to May 's dismay and annoyance . Eventually , the three reach Bingo with May 's fish dead , Clarkson 's fish ruined and Hammond having almost no fish left . As a result , they sell no fish and leave Bingo in a helicopter `` like all charitable celebrities '' .
  Fact 0.0: Episode 14 is titled Past, Present, or Future.
  Fact 0.1: Episode 14 is numbered 14 overall and 14 in the series.
  Fact 0.2: Episode 14 tests the Lamborghini Aventador S, Honda NSX, and Rimac Concept One.
  Fact 0.3: Episode 14 has special guests David Hasselhoff and Ricky Wilson.
  Fact 0.4: Episode 14 originally aired on 8 December 2017.
  Fact 0.5: The tent is now in the Cotswolds and will remain there for the duration of the series.
  Fact 0.6: The presenters want to determine whether the past, present, or future of motoring is best.
  Fact 0.7: The presenters visit Switzerland with Clarkson in a petrol powered Lamborghini Aventador S representing the past.
  Fact 0.8: May drives a hybrid powered Honda NSX representing the present.
  Fact 0.9: Hammond drives an all-electric Rimac Concept One representing the future.
  Fact 0.10: After starting at the top of the Gotthard Pass, Hammond brings the trio to a stay at the Park Weggis (de) health retreat.
  Fact 0.11: Hammond visits the Swiss Museum of Transport and other museums in Lucerne.
  Fact 0.12: The Park Weggis stay is close to a charging station for the Rimac car.
  Fact 0.13: May and Clarkson reject the plan and go to a hill climbing race at Hemberg instead.
  Fact 0.14: Clarkson cannot fit into the Aventador whilst wearing a mandatory racing helmet.
  Fact 0.15: Clarkson sends a Lamborghini test driver in his place.
  Fact 0.16: The Lamborghini test driver sets the fastest time of all three.
  Fact 0.17: Hammond crashes after completing the course, destroying the Rimac in a fire.
  Fact 0.18: Footage taken from YouTube shows the wreckage being extinguished.
  Fact 0.19: Footage taken from YouTube shows Hammond being airlifted by the Swiss Air-Rescue (REGA) to hospital.
  Fact 0.20: The presenters discuss some inappropriate new features within cars on Conversation Street.
  Fact 0.21: Celebrity Brain Crash is replaced by Celebrity Face Off.
  Fact 0.22: In Celebrity Face Off, Ricky Wilson beats David Hasselhoff to determine who the fastest former talent show host is.
  Fact 0.23: Episode 15 is titled The Falls Guys.
  Fact 0.24: Episode 15 is numbered 15 overall and 15 in the series.
  Fact 0.25: Episode 15 tests the Ford GT and Mercedes AMG GT R.
  Fact 0.26: Episode 15 has special guests Kevin Pietersen and Brian Wilson.
  Fact 0.27: Episode 15 originally aired on 15 December 2017.
  Fact 0.28: The presenters race from Central Park in Manhattan, New York City, to an observation tower overlooking the Niagara Falls.
  Fact 0.29: Clarkson drives the new Ford GT.
  Fact 0.30: May and Hammond use public transport, with Hammond on crutches.
  Fact 0.31: May and Hammond take the New York City Subway, AirTrain JFK, a JetBlue Embraer E190 flight, and two Niagara Frontier Transportation Authority buses.
  Fact 0.32: Hammond and May's flight arrives 15 minutes early.
  Fact 0.33: Clarkson narrowly wins the race.
  Fact 0.34: For Conversation Street the presenters discuss sponsorship ideas.
  Fact 0.35: Clarkson reviews the Mercedes AMG GT R.
  Fact 0.36: After the review, new test driver Abbie Eaton takes the Mercedes AMG GT R for a timed lap around the Eboladrome.
  Fact 0.37: In Celebrity Face Off, Kevin Pietersen of South Africa beats Brian Wilson of the U.S. as the fastest person who makes a living throwing and hitting and catching small balls.
  Fact 0.38: Episode 16 is titled Bah Humbug - atti.
  Fact 0.39: Episode 16 is numbered 16 overall and 16 in the series.
  Fact 0.40: Episode 16 tests the Kia Stinger GT and Bugatti Chiron.
  Fact 0.41: Episode 16 has special guests Hugh Bonneville and Casey Anderson.
  Fact 0.42: Episode 16 originally aired on 22 December 2017.
  Fact 0.43: The presenters begin with Christmas tree lights made from vehicle parts.
  Fact 0.44: Starting with Scalextric, the presenters exchange humorous and inappropriate Christmas gifts in Conversation Street.
  Fact 0.45: May travels to Majorca to race a Kia Stinger GT against a pair of longboard riders along a mountain road.
  Fact 0.46: May wins the race despite him racing uphill and the riders downhill.
  Fact 0.47: Hammond and May attempt to relieve lunchtime boredom by inventing lunchtime office car park racing at Houghton Hall business park in the Luton / Dunstable Urban Area.
  Fact 0.48: After brief interviews, Hugh Bonneville and Casey Anderson each share their own ownership of a Tesla Model S.
  Fact 0.49: Hugh Bonneville represents the UK, and Casey Anderson represents the United States.
  Fact 0.50: Hugh Bonneville and Casey Anderson compete during Celebrity Face Off as the fastest person who lives with a bear.
  Fact 0.51: Casey Anderson wins by setting the faster lap time.
  Fact 0.52: Clarkson experiments with the jet set lifestyle.
  Fact 0.53: Clarkson starts the morning from the Da Vinci super-yacht moored in Saint-Tropez.
  Fact 0.54: Clarkson then drives a Bugatti Chiron to L'Alpe De Venosc France for skiing at lunchtime.
  Fact 0.55: Clarkson stops briefly for a drag race against a BMW M4 at Gap -- Tallard Airport to give viewers a perspective on how fast the Chiron is.
  Fact 0.56: After leaving the Alps, Clarkson pays an homage to the 1976 film C'etait un rendez-vous.
  Fact 0.57: Clarkson speeds across Turin and reaches the Piazza Carignano (it) in 9 minutes.
  Fact 0.58: Episode 17 is titled Unscripted.
  Fact 0.59: Episode 17 is numbered 17 overall and 17 in the series.
  Fact 0.60: Episode 17 tests the McLaren 720S, Audi TT RS, and Ariel Nomad.
  Fact 0.61: Episode 17 has special guests Michael Ball, Alfie Boe, and Silvija Jurin.
  Fact 0.62: Episode 17 originally aired on 29 December 2017.
  Fact 0.63: The presenters begin by having a year-ending awards ceremony in Conversation Street.
  Fact 0.64: Hammond heads to the Eboladrome to test the McLaren 720S.
  Fact 0.65: Eaton sets a lap time after Hammond tests the McLaren 720S.
  Fact 0.66: The Grand Tour has been criticised for being too scripted.
  Fact 0.67: The presenters head to Zadar county and the island of Pag in Croatia to make a completely unscripted film.
  Fact 0.68: Clarkson turns up in the Audi TT RS.
  Fact 0.69: Hammond turns up in the not at all comparable Ariel Nomad.
  Fact 0.70: May turns up in an old Lada Riva which he turns into a fire engine.
  Fact 0.71: The presenters go their separate ways.
  Fact 0.72: Clarkson and Hammond end up comparing their cars in a drag race and a rally stage, both of which are won by Clarkson.
  Fact 0.73: Clarkson and Hammond visit Maslenica, the Plitvice Lakes National Park, and follow the D106 road until running out of road at Zigljen ferry port.
  Fact 0.74: Due to a lack of fires, May has to set up his own ones.
  Fact 0.75: Clarkson and Hammond set up a fire that proves too difficult for May.
  Fact 0.76: The real fire brigade puts out the fire with a firefighting aircraft.
  Fact 0.77: The firefighting aircraft destroys May's Lada fire engine in the process.
  Fact 0.78: In Celebrity Face Off, Michael Ball sets the faster time against Alfie Boe.
  Fact 0.79: Michael Ball becomes the fastest classical singer with some kind of interesting connection to the British motor industry.
  Fact 0.80: Episode 18 is numbered 5 in the series and is titled Up, Down and Round the Farm.
  Fact 0.81: Episode 18 tests the Ripsaw EV2 and Volkswagen Up! GTI.
  Fact 0.82: Episode 18 has special guests Bill Bailey, Dominic Cooper, and Mark Higgins.
  Fact 0.83: Episode 18 originally aired on 5 January 2018.
  Fact 0.84: The presenters introduce some of the new cars for 2018 in Conversation Street.
  Fact 0.85: Hammond heads to Dubai to test a Sandrail dune buggy in the desert.
  Fact 0.86: Hammond then tests the Ripsaw high-end luxury tank.
  Fact 0.87: The presenters visit the Dubai Mall.
  Fact 0.88: May reviews the Volkswagen Up! GTI at the Eboladrome.
  Fact 0.89: In thick fog, Eaton records a record slow time.
  Fact 0.90: Guests Bill Bailey and Dominic Cooper discuss their shared ownership of old French cars.
  Fact 0.91: Bill Bailey and Dominic Cooper race each other in greasy and wet conditions.
  Fact 0.92: Dominic Cooper becomes the fastest person who has a failed former career in a band.
  Fact 0.93: Clarkson creates a Ken Block-style Farmkhana film using an old Subaru Impreza WRX STi around Hare Farm, near Rye.
  Fact 0.94: The Farmkhana film uses bullet time photography.
  Fact 0.95: Hammond and May show the audience that Clarkson's driving was mostly fake.
  Fact 0.96: They share a making-of consisting of outtakes and behind-the-scenes footage.
  Fact 0.97: The making-of features rally driver Mark Higgins, producer Phil Churchward, and camera operator Ben Joiner breaking the fourth wall.
  Fact 0.98: Clarkson suggests that the whole thing is like sawing a woman in half during a magic show.
  Fact 0.99: Clarkson suggests the audience do not really want to know.
  Fact 0.100: Episode 19 is numbered 6 in the series and is titled Jaaaaaaaags.
  Fact 0.101: Episode 19 has no vehicles tested.
  Fact 0.102: Episode 19 has special guests Luke Evans and Kiefer Sutherland.
  Fact 0.103: Episode 19 originally aired on 12 January 2018.
  Fact 0.104: The presenters head to the area around Grand Junction, Colorado, staying at the Gateway Canyons Resort & Spa.
  Fact 0.105: The presenters travel to Telluride, Colorado with a variety of old Jaguar cars.
  Fact 0.106: The trip establishes the manufacturer's reputation for reliability.
  Fact 0.107: Clarkson buys an XJR, May takes an XK8 convertible, and Hammond brings a Mark X.
  Fact 0.108: The cars are driven around a dirt track, where Hammond's car is badly damaged and is replaced by an XJ6.
  Fact 0.109: The cars are driven along the runway at Telluride Regional Airport, where May's car is destroyed and is replaced by an XJS.
  Fact 0.110: The cars are driven down the slopes of the Telluride Ski Resort.
  Fact 0.111: At Hanging Flume overlook, Clarkson experiences a traffic stop by a Mesa County police officer.
  Fact 0.112: Clarkson reads the Her Britannic Majesty text from his British passport.
  Fact 0.113: Despite Clarkson's car being the best, May admits his car had poor brakes.
  Fact 0.114: The presenters deem the challenge pointless.
  Fact 0.115: The presenters discuss the annoyance of Firemen cutting the roofs of cars in Conversation Street.
  Fact 0.116: Kiefer Sutherland beats Luke Evans to become the fastest actor with a middle name of George.
  Fact 0.117: Episode 20 is numbered 7 in the series and is titled It's a Gas, Gas, Gas.
  Fact 0.118: Episode 20 tests the Lamborghini Huracan Performante.
  Fact 0.119: Episode 20 has special guests Anthony Joshua and Bill Goldberg.
  Fact 0.120: Episode 20 originally aired on 19 January 2018.
  Fact 0.121: The presenters discuss new supercars and how handbrake turns impress women within the Conversation Street segment.
  Fact 0.122: Hammond tests the Lamborghini Huracan Performante at the Eboladrome.
  Fact 0.123: Hammond beats May's Ferrari 458 Speciale in a drag race.
  Fact 0.124: Eaton takes the Lamborghini Huracan Performante round the track and sets a fast time.
  Fact 0.125: May and Hammond look at ingenious ways of filling up with fuel instead of stopping at a fuel station.
  Fact 0.126: Anthony Joshua becomes the fastest man who makes a living out of punching or strangling someone.
  Fact 0.127: Anthony Joshua beats Bill Goldberg.
  Fact 0.128: Anthony Joshua and Bill Goldberg both previously owned cars from General Motors.
  Fact 0.129: Clarkson talks about the rivalry in rallying between Audi with the Quattro and Lancia with the 037.
  Fact 0.130: The presenters conclude with a terrible disappointment that Lancia is no longer around.
  Fact 0.131: Episode 21 is numbered 8 in the series and is titled Blasts from the Past.
  Fact 0.132: Episode 21 tests the Jaguar XK SS, Aston Martin DB4 GT, Honda Civic Type R, and Ford GT.
  Fact 0.133: Episode 21 has special guests Stewart Copeland and Nick Mason.
  Fact 0.134: Episode 21 originally aired on 26 January 2018.
  Fact 0.135: The presenters head to the City of Pau, Pyrenees-Atlantiques, France, home of the Pau Grand Prix on the Circuit de Pau-Ville (fr) street course.
  Fact 0.136: At Palais Beaumont (fr), Hammond arrives with a Jaguar XK SS.
  Fact 0.137: Clarkson arrives at Palais Beaumont (fr) in an Aston Martin DB4 GT.
  Fact 0.138: Both cars are modern builds of older designs from the 1950s back-catalogues of Jaguar and Aston Martin.
  Fact 0.139: Hammond and Clarkson race around the Streets of Pau.
  Fact 0.140: May turns up in a modern Honda Civic Type R to prove that new cars are better.
  Fact 0.141: All three drive south over the Pyrenees mountain range.
  Fact 0.142: Clarkson and Hammond experience difficulties in dense fog.
  Fact 0.143: Clarkson recounts a 2013 incident in which Clarkson and Hammond's driving licences were confiscated whilst filming The Perfect Road Trip in France.
  Fact 0.144: Clarkson tries to goad May into speeding on the Autopista AP-2.
  Fact 0.145: At Barcelona, the presenters race around the abandoned Autodromo de Sitges-Terramar and its extreme banking on its corners.
  Fact 0.146: May wins the race at Autodromo de Sitges-Terramar.
  Fact 0.147: Hammond abstains from the race following his history of crashes.
  Fact 0.148: During Conversation Street the presenters discuss names for gritters.
  Fact 0.149: At the Eboladrome, Clarkson retests the Ford GT following the Niagara Falls race.
  Fact 0.150: Eaton sets a test lap.
  Fact 0.151: Nick Mason records a faster lap than Stewart Copeland.
  Fact 0.152: Nick Mason becomes the fastest drummer for a band beginning with the letter P.
  Fact 0.153: Episode 22 is numbered 9 in the series and is titled Breaking, Badly.
  Fact 0.154: Episode 22 tests the Jaguar XJ220 and Bugatti EB 110 Super Sport.
  Fact 0.155: Episode 22 has special guests Dynamo and Penn & Teller.
  Fact 0.156: Episode 22 originally aired on 2 February 2018.
  Fact 0.157: The presenters discuss new sport cars by various companies, jumpers, and the speed of pigeons vs blackbirds during Conversation Street.
  Fact 0.158: Clarkson unsuccessfully sets out to prove that mankind has not significantly improved over the last 25 years.
  Fact 0.159: The Jaguar XJ220 and Bugatti EB 110 Super Sport are the fastest supercars from the early 90s.
  Fact 0.160: Both the Jaguar XJ220 and Bugatti EB 110 Super Sport set slow times at the Eboladrome when driven by Eaton.
  Fact 0.161: The trio tries to break the UK speed record for amphibious vehicles by building one of their own.
  Fact 0.162: Clarkson oversees his idea of combining a Rolls-Royce Spey turbofan engine from a Nimrod jet and a speedboat with a Suzuki Jimny.
  Fact 0.163: On a nearby lake, Clarkson and May have to abort their run after achieving only 3 mph (5 km/h) and nearly sinking.
  Fact 0.164: Hammond and May build a vehicle by combining a Bond Bug with the engine of a personal water craft.
  Fact 0.165: They nickname the vehicle the Pond Bug.
  Fact 0.166: Competing on Coniston Water during Coniston Power Boat Records Week, Clarkson operates the Pond Bug.
  Fact 0.167: Clarkson is the only one with a speedboat licence.
  Fact 0.168: Clarkson achieves an average of 47.81 mph (76.94 km/h), setting a new UK record.
  Fact 0.169: After performing a series of magic tricks, Dynamo and Penn & Teller compete to become the fastest magician in the world.
  Fact 0.170: Penn & Teller prevail on a snowy track.
  Fact 0.171: Episode 23 is numbered 10 in the series and is titled Oh, Canada.
  Fact 0.172: Episode 23 tests the Alfa Romeo Stelvio Quadrifoglio, Range Rover Velar P380, Porsche Macan Turbo Performance Pack, Tesla Model X, and Ford F150 Raptor.
  Fact 0.173: Episode 23 has special guests Rory McIlroy and Paris Hilton.
  Fact 0.174: Episode 23 originally aired on 9 February 2018.
  Fact 0.175: The presenters head to Canada to prove that small SUVs are useless.
  Fact 0.176: Clarkson picks an Alfa Romeo Stelvio Quadrifoglio.
  Fact 0.177: May chooses a Range Rover Velar P380.
  Fact 0.178: Hammond takes a Porsche Macan Turbo Performance Pack.
  Fact 0.179: The SUVs are shown to be slower round the Area 27 circuit than a BMW M3.
  Fact 0.180: The SUVs are not practical enough to carry Canadian Newfoundland dogs.
  Fact 0.181: The SUVs are slower round a barrel course than a horse.
  Fact 0.182: The presenters conclude that SUVs are pointless.
  Fact 0.183: Executive producer Andy Wilman informs via text that they will race their SUVs whilst each towing a boat up a mountain.
  Fact 0.184: May abandons his Velar for his boat, which sinks, causing him to lose the race.
  Fact 0.185: Clarkson's Stelvio has no tow bar, so he cheats and exchanges it for a Ford F150 Raptor.
  Fact 0.186: Clarkson arrives first in the Ford F150 Raptor.
  Fact 0.187: The presenters conclude that the Raptor was better than all the SUVs being tested.
  Fact 0.188: On Conversation Street the presenters talk about new SUVs and vans.
  Fact 0.189: Clarkson reviews the Tesla Model X at the Eboladrome and on the road.
  Fact 0.190: Clarkson is accompanied by some lawyers to avoid Tesla pursuing any legal action.
  Fact 0.191: The lawyers follow the previous Tesla v. Top Gear court cases.
  Fact 0.192: Rory McIlroy and Paris Hilton discuss their car collections.
  Fact 0.193: McIlroy beats Hilton to become the World's Fastest Golfing Enthusiast.
  Fact 0.194: Episode 24 is numbered 11 in the series and is titled Feed the World.
  Fact 0.195: Episode 24 has no vehicles tested and no special guests.
  Fact 0.196: Episode 24 originally aired on 16 February 2018.
  Fact 0.197: The presenters head to Mozambique on a mission to feed the people of Bingo, an isolated town in the west of the country.
  Fact 0.198: The fish come from the country's capital of Maputo.
  Fact 0.199: The fish are carried a distance of approximately 200 miles.
  Fact 0.200: May buys a Mercedes-Benz Estate and fits a Perspex tank filled with sea water in which to keep the fish alive.
  Fact 0.201: Clarkson buys a Nissan Hardbody truck which he fits with a generator and an ice machine to keep the fish frozen.
  Fact 0.202: Hammond buys a TVS Star motorcycle and devises a rack which he fits to the bike's rear.
  Fact 0.203: The rack is inspired by a local method of drying and salting the fish to preserve them.
  Fact 0.204: After unsuccessfully trying to catch fish using local methods, they buy fish at a local market.
  Fact 0.205: Early in the journey, Clarkson's truck crashes into May's car upon rescuing it, damaging its radiator grille.
  Fact 0.206: Deeper into rural Mozambique, May constantly soaks himself with sea water from his ill-fitted tank.
  Fact 0.207: Due to his damaged radiator grille, May suffers multiple breakdowns due to water intrusion.
  Fact 0.208: Hammond loses a large number of his fish on the bumpy roads.
  Fact 0.209: Hammond constantly gets stuck in thick mud and endlessly falls from his motorbike.
  Fact 0.210: Clarkson suffers very few issues except when his ice machine breaks.
  Fact 0.211: Clarkson makes small adjustments to his truck in order to smoke the fish with his exhaust.
  Fact 0.212: When May's car breaks down in the middle of a lake, Hammond and Clarkson cross successfully.
  Fact 0.213: Clarkson cuts off the front end of the Mercedes and tows what remains of the rear portion using his truck, much to May's dismay and annoyance.
  Fact 0.214: The three reach Bingo with May's fish dead.
  Fact 0.215: Clarkson's fish are ruined.
  Fact 0.216: Hammond has almost no fish left.
  Fact 0.217: As a result, they sell no fish.
  Fact 0.218: They leave Bingo in a helicopter like all charitable celebrities.

Retrieved items:
<item 1>
The Grand Tour (TV series)
The other two are so intoxicated by their own debate they do n't notice . 2.10 It 's time to step in a dog turd of chat , on Conversation Street . The stools
have disappeared . Clarkson with a sledge hammer in his hands , Hammond with a shovel are standing above a third person ( May ? ) They are swinging their tools
at the prone body which cries out in pain when it is hit . ( this is the only time in two series that we hear any one of the three make a noise in an otherwise
silent introduction ) Running gags ( edit ) The first series included several running gags . During the opening titles of each episode , a camera drone was
destroyed . From episode three onward , one of the presenters ' names is always seen misspelt during the opening . For `` Celebrity Brain Crash '' , celebrity
guests are humorously `` killed '' in an accident while making their way to the tent . After each `` death '' , May asks `` Does that mean he / she 's not
coming on then ? ''
</item 1>

<item 2>
The Grand Tour (TV series)
Studio audience segments for the first series were filmed in the travelling tent between 17 July and 11 December 2016 in various countries , while for the
second series the tent was stationed permanently in the Cotswolds . Series one was filmed over the course of mid - to late 2016 and premiered on 18 November
2016 . Series two was filmed over the same time period the following year ; however , filming was set back due to various illnesses and injuries suffered by the
presenters . Due to this , it premiered on 8 December 2017 . In the days following the launch , `` The Holy Trinity '' became Amazon Video 's most watched
premiere episode . Overall , the show has received positive reviews from critics . Contents ( hide ) 1 Format 1.1 Films 1.2 Test tracks 1.2. 1 Eboladrome 1.2. 2
Enstone 1.3 Celebrity Face Off 1.4 Conversation Street 1.5 Running gags 1.6 Awards 2 Episodes 2.1 Series overview 2.2 Series 1 ( 2016 -- 17 ) 2.2.
</item 2>

<item 3>
Graph query row:
episode: The Falls Guys
relationship: RELATED_TO
related: Kevin Pietersen
</item 3>

<item 4>
The Grand Tour (TV series)
ASIN B0731ZQ3R7. 1 . Past , present or future Missing or empty url = ( help ) ; access - date = requires url = ( help ) ^ Jump up to : Starkey , Adam ( 13
December 2017 ) . `` The Grand Tour season 2 : Richard Hammond is on crutches in episode 2 after horrifying crash '' . Metro . London . Retrieved 13 December
2017 . second episode titled The Falls Guys , Clarkson drives ... Ford GT from New York City to Niagara Falls , with James May making the same journey via
public transport with Hammond ... Celebrity Face Off ... cricketer Kevin Pietersen and baseball 's Brian Wilson Jump up ^ `` The Grand Tour Season 2 Episode 3
'' . 16 December 2017 . Retrieved 16 December 2017 . ^ Jump up to : `` Zadarska manekenka Silvija oduševila u ' The Grand Touru ' '' ( Zadar model Silvija
delights in The Grand Tour ) . 24sata ( in Croatian ) . 3 January 2018 .
</item 4>

<item 5>
The Grand Tour (TV series)
The Daily Telegraph , while not as positive , still approved of the episode , stating `` The writing is still rather ropey . Clarkson 's suggestion of a new
nickname for May -- `` Dingleberry Handpump '' -- failed to raise a titter even among the super-fans gathered for the London premiere '' but also said that ``
for each wobble , there are just as many moments when The Grand Tour manages the clever trick Top Gear could pull off at its best : raising a chuckle while
sneaking in a bit of serious journalism at the same time . '' and ultimately gave the episode 3 out of 5 stars . Jeremy Clarkson himself believed that they had
`` hit the ground running with series 2 of the Grand Tour '' . Radio Times has criticised the gender balance of guests on the `` Celebrity Face Off '' segment .
There was only 1 woman guest in the episode `` Oh Canada '' ; Paris Hilton . Hothouse Flowers playing for the opening sequence of `` The Holy Trinity '' episode
in the Lucerne Valley , California Broadcast ( edit ) Australian free - to - air network Seven Network started broadcasting the first series of The Grand Tour
in mid-October 2017 .
</item 5>

<item 6>
The Grand Tour (TV series)
The Sun . Retrieved 23 December 2017 . ^ Jump up to : Daly , Emma ( 26 November 2016 ) . `` Jeremy Clarkson forced to ' kill off ' celebs on The Grand Tour
after BBC ' bans Top Gear rip - off star interviews ' '' . The Sun . Retrieved 26 November 2016 . Jump up ^ Lee , Kristen ( 16 November 2017 ) . `` ' Celebrity
Brain Crash ' Axed on the Grand Tour Season 2 '' . jalopnik.com . Retrieved 17 November 2017 . Jump up ^ McCreesh , Louise ( 27 November 2017 ) . `` The Grand
Tour season 2 confirms celebrity guests including The Hoff , Jack Bauer and Luke Evans '' . Digital Spy . Retrieved 28 November 2017 . Jump up ^ Hsieh , Tony (
20 November 2016 ) . `` The Grand Tour : 3 Mustangs from the Opening Sequence of the Premiere Episode '' . Grand Tour Nation . Retrieved 24 November 2016 . ^
Jump up to : `` Amazon Instant Video '' .
</item 6>

<item 7>
The Grand Tour (TV series)
The Sunday Times : Driving . Retrieved 14 December 2017 . The show has a new test track in Enstone , ... there planning problems when ( Clarkson ) tried once
before to use the Enstone track , back in Top Gear days ... The noisier testing of fast cars will continue to be done in Wiltshire at RAF Wroughton ...
nicknamed `` the Eboladrome '' Jump up ^ Reed , Jason ( 24 January 2018 ) . `` The Secret Story Behind The Grand Tour 's Celebrity Braincrash '' . Shifting
Lanes . Retrieved 26 January 2018 . ^ Jump up to : McCreesh , Louise ( 27 November 2017 ) . `` The Grand Tour season 2 confirms celebrity guests including
Kiefer Sutherland and David Hasselhoff '' . Retrieved 19 December 2017 . The celebrity guests this series include Luke Evans , Kiefer Sutherland , Hugh
Bonneville , Kevin Pietersen and Dominic Cooper ( ... ) Dynamo , Rory Mcllroy , Michael Ball and Alfie Boe will also appear in season two .
</item 7>

<item 8>
Graph query row:
episode: The Falls Guys
relationship: RELATED_TO
related: Brian Wilson
</item 8>

<item 9>
The Grand Tour (TV series)
1 Notes 2.3 Series 2 ( 2017 -- 18 ) 2.3. 1 Notes 2.4 Series 3 3 Production 3.1 History 3.2 Legal issues 3.3 Filming 3.4 Promotion 3.5 Sponsorship 3.6 Technical
Issues 4 Reception 5 Broadcast 6 See also 7 References 8 External links Format ( edit ) The Grand Tour is presented as a mix of pre-recorded television films
and live - audience segments presented in a large tent to an audience of about 300 members . The show and its segments are hosted by Jeremy Clarkson , Richard
Hammond , and James May . The show is presented from a pair of large tents on a Grand Tour around the world . The presenters sit around a trestle table , with
the studio audience seated in front of them . Films ( edit ) Each show is generally bookended by two television films , or two parts of the same film , produced
by one or more of the hosts .
</item 9>

<item 10>
The Grand Tour (TV series)
Rotten Tomatoes . Retrieved 17 July 2017 . Jump up ^ `` The Grand Tour season 2 episode 1 review : We 're in safe hands as Clarkson , Hammond and May return ''
. Digital Spy . Retrieved 8 December 2017 . Jump up ^ `` Television review : The Grand Tour on Amazon Prime '' . The Times . Retrieved 8 December 2017 . Jump up
^ `` The Grand Tour season 2 , episode 1 review : Has Jeremy Clarkson finally found his groove ? '' . The Daily Telegraph . Retrieved 8 December 2017 . Jump up
^ Clarkson , Jeremy ( 8 December 2017 ) . `` Jeremy Clarkson on Twitter '' . twitter.com . Retrieved 15 December 2017 . Jump up ^ Ling , Thomas ( 27 November
2017 ) . `` The Grand Tour 's first celebrity guests for series 2 have been revealed -- and there 's not a single woman '' . Retrieved 20 December 2017 . Jump
up ^ Long , Trevor .
</item 10>
=== END ITEM id=1324176cb8596335 ===
